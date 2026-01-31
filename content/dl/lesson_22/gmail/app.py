import os
import pickle
import base64
import torch
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build


# CONFIG
SCOPES = ["https://www.googleapis.com/auth/gmail.modify"]
MODEL_NAME = "Sanjarbek1024/smile-text-classifier"

EMOTIONS = [
    "angry", "disgust", "happy",
    "not-relevant", "sad", "surprise"
]


# MODEL
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()

id2label = {
    0: "angry",
    1: "disgust",
    2: "happy",
    3: "not-relevant",
    4: "sad",
    5: "surprise"
}
model.config.id2label = id2label


def predict_emotion(text):
    inputs = tokenizer(
        text, return_tensors="pt",
        truncation=True, padding=True, max_length=512
    )

    with torch.no_grad():
        logits = model(**inputs).logits

    probs = F.softmax(logits, dim=1)
    idx = torch.argmax(probs, dim=1).item()
    return id2label[idx], probs[0][idx].item()


# Gmail Auth
def gmail_auth():
    if os.path.exists("token.pickle"):
        with open("token.pickle", "rb") as f:
            creds = pickle.load(f)
    else:
        flow = InstalledAppFlow.from_client_secrets_file(
            "credentials.json", SCOPES
        )
        creds = flow.run_local_server(port=0)
        with open("token.pickle", "wb") as f:
            pickle.dump(creds, f)

    return build("gmail", "v1", credentials=creds)


def get_all_messages(service):
    messages = []
    page_token = None

    while True:
        res = service.users().messages().list(
            userId="me",
            labelIds=["INBOX"],
            pageToken=page_token,
            maxResults=500
        ).execute()

        messages.extend(res.get("messages", []))
        page_token = res.get("nextPageToken")

        if not page_token:
            break

    return messages


def get_text(service, msg_id):
    msg = service.users().messages().get(
        userId="me", id=msg_id, format="full"
    ).execute()

    text = ""
    for h in msg["payload"]["headers"]:
        if h["name"] == "Subject":
            text += h["value"] + " "

    for part in msg["payload"].get("parts", []):
        if part["mimeType"] == "text/plain":
            text += base64.urlsafe_b64decode(
                part["body"]["data"]
            ).decode("utf-8")

    return text.strip()


def get_or_create_label(service, name):
    labels = service.users().labels().list(
        userId="me"
    ).execute()["labels"]

    for l in labels:
        if l["name"].lower() == name.lower():
            return l["id"]

    return service.users().labels().create(
        userId="me",
        body={
            "name": name.capitalize(),
            "labelListVisibility": "labelShow",
            "messageListVisibility": "show"
        }
    ).execute()["id"]


def move(service, msg_id, label_id):
    service.users().messages().modify(
        userId="me",
        id=msg_id,
        body={
            "addLabelIds": [label_id],
            "removeLabelIds": ["INBOX"]
        }
    ).execute()


# MAIN
def main():
    service = gmail_auth()

    label_ids = {
        e: get_or_create_label(service, e)
        for e in EMOTIONS
    }

    messages = get_all_messages(service)
    print("Total inbox emails:", len(messages))

    for m in messages:
        text = get_text(service, m["id"])
        if not text:
            continue

        label, conf = predict_emotion(text)
        if conf < 0.6:
            label = "not-relevant"

        move(service, m["id"], label_ids[label])


if __name__ == "__main__":
    main()
