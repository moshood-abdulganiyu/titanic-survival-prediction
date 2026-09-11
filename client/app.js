"use strict";

const form = document.getElementById("predictionForm");
const resultPanel = document.getElementById("resultPanel");
const resultHeadline = document.getElementById("resultHeadline");
const resultDetail = document.getElementById("resultDetail");
const iconSurvived = document.querySelector(".icon-survived");
const iconNotSurvived = document.querySelector(".icon-not-survived");
const submitButton = form.querySelector(".submit-btn");
const submitButtonDefaultText = submitButton.textContent;

// Local FastAPI dev server from Step 4/5. This is the one line to
// change when Step 8 points the frontend at the deployed HF Space.
const API_URL = "http://127.0.0.1:8000/predict";

// Friendlier validity messages, matching the PassengerInput bounds in
// server/schemas.py. Kept in one place so a schema change only needs
// an edit here, not a hunt through the markup.
const VALIDATION_MESSAGES = {
  Pclass: "Select a ticket class.",
  Sex: "Select a sex.",
  Age: "Enter an age between 0 and 100.",
  Fare: "Enter a fare of 0 or more.",
  SibSp: "Enter a number between 0 and 8.",
  Parch: "Enter a number between 0 and 6.",
  Embarked: "Select a port of embarkation.",
};

function applyCustomValidity(field) {
  if (field.validity.valid) {
    field.setCustomValidity("");
    return;
  }
  field.setCustomValidity(VALIDATION_MESSAGES[field.name] || "Check this field.");
}

// Clear a stale custom message as soon as the person edits the field,
// otherwise the browser keeps showing the old bubble even after a fix.
for (const field of form.elements) {
  if (!field.name) continue;
  field.addEventListener("invalid", () => applyCustomValidity(field));
  field.addEventListener("input", () => field.setCustomValidity(""));
  field.addEventListener("change", () => field.setCustomValidity(""));
}

function buildPayload() {
  const data = new FormData(form);
  return {
    Pclass: Number(data.get("Pclass")),
    Sex: data.get("Sex"),
    Age: Number(data.get("Age")),
    SibSp: Number(data.get("SibSp")),
    Parch: Number(data.get("Parch")),
    Fare: Number(data.get("Fare")),
    Embarked: data.get("Embarked"),
  };
}

function showResult(result) {
  const survived = result.survived === 1;

  resultPanel.hidden = false;
  resultPanel.classList.remove("is-error");
  resultPanel.classList.toggle("is-survived", survived);
  resultPanel.classList.toggle("is-not-survived", !survived);

  iconSurvived.hidden = !survived;
  iconNotSurvived.hidden = survived;

  resultHeadline.textContent = survived ? "Survived" : "Did not survive";
  resultDetail.textContent =
    "Estimated probability: " + (result.probability * 100).toFixed(1) + "%";
}

function showError(message) {
  resultPanel.hidden = false;
  resultPanel.classList.remove("is-survived", "is-not-survived");
  resultPanel.classList.add("is-error");

  iconSurvived.hidden = true;
  iconNotSurvived.hidden = true;

  resultHeadline.textContent = "Couldn't get a prediction";
  resultDetail.textContent = message;
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  for (const field of form.elements) {
    if (field.name) applyCustomValidity(field);
  }

  if (!form.reportValidity()) return;

  const payload = buildPayload();

  submitButton.disabled = true;
  submitButton.textContent = "Checking...";

  try {
    const response = await fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      // FastAPI validation errors (422) come back with a `detail` field;
      // anything else just gets the status text.
      const body = await response.json().catch(() => null);
      const detail = body && body.detail ? JSON.stringify(body.detail) : response.statusText;
      throw new Error(`Server returned ${response.status}: ${detail}`);
    }

    const result = await response.json();
    showResult(result);
  } catch (error) {
    console.error("Prediction request failed:", error);
    showError(
      "Couldn't reach the prediction server. Confirm the FastAPI backend is running on " +
        API_URL +
        " and try again."
    );
  } finally {
    submitButton.disabled = false;
    submitButton.textContent = submitButtonDefaultText;
  }
});
