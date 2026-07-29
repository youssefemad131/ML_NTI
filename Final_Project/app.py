"""
Hotel Booking Cancellation Predictor - Gradio App
--------------------------------------------------
Loads `hotel_cancellation_pipeline.pkl` (produced by the notebook's
"Production Pipeline + Grid Search" section) and serves a small web form
to predict whether a booking will be canceled.

Run:
    conda activate <your_env>
    pip install -r requirements.txt
    python app.py

Then open the local URL Gradio prints (usually http://127.0.0.1:7860).
"""

import joblib
import pandas as pd
import gradio as gr

MODEL_PATH = "hotel_cancellation_pipeline.pkl"
model = joblib.load(MODEL_PATH)

DEPOSIT_TYPES = ["No Deposit", "Non Refund", "Refundable"]
CUSTOMER_TYPES = ["Transient", "Contract", "Transient-Party", "Group"]
MARKET_SEGMENTS = [
    "Direct", "Corporate", "Online TA", "Offline TA/TO",
    "Complementary", "Groups", "Undefined", "Aviation",
]


def predict_cancellation(
    lead_time,
    stays_in_weekend_nights,
    stays_in_week_nights,
    adults,
    booking_changes,
    adr,
    total_of_special_requests,
    required_car_parking_spaces,
    previous_bookings_not_canceled,
    deposit_type,
    customer_type,
    market_segment,
):
    row = pd.DataFrame([{
        "lead_time": lead_time,
        "stays_in_weekend_nights": stays_in_weekend_nights,
        "stays_in_week_nights": stays_in_week_nights,
        "adults": adults,
        "booking_changes": booking_changes,
        "adr": adr,
        "total_of_special_requests": total_of_special_requests,
        "required_car_parking_spaces": required_car_parking_spaces,
        "previous_bookings_not_canceled": previous_bookings_not_canceled,
        "deposit_type": deposit_type,
        "customer_type": customer_type,
        "market_segment": market_segment,
    }])

    pred = model.predict(row)[0]
    proba = model.predict_proba(row)[0][1]  # probability of class "1" = Canceled

    label = "❌ Likely to Cancel" if pred == 1 else "✅ Likely to Show Up"
    return f"{label}\n\nCancellation probability: {proba:.1%}"


with gr.Blocks(title="Hotel Booking Cancellation Predictor") as demo:
    gr.Markdown("# 🏨 Hotel Booking Cancellation Predictor")
    gr.Markdown(
        "Fill in the booking details below and click **Predict** to estimate "
        "the probability that this booking will be canceled."
    )

    with gr.Row():
        with gr.Column():
            lead_time = gr.Number(label="Lead Time (days before arrival)", value=30)
            stays_in_weekend_nights = gr.Number(label="Weekend Nights", value=1)
            stays_in_week_nights = gr.Number(label="Week Nights", value=2)
            adults = gr.Number(label="Adults", value=2)
            booking_changes = gr.Number(label="Booking Changes", value=0)
            adr = gr.Number(label="Average Daily Rate (ADR)", value=100.0)

        with gr.Column():
            total_of_special_requests = gr.Number(label="Total Special Requests", value=0)
            required_car_parking_spaces = gr.Number(label="Required Parking Spaces", value=0)
            previous_bookings_not_canceled = gr.Number(label="Previous Bookings Not Canceled", value=0)
            deposit_type = gr.Dropdown(DEPOSIT_TYPES, label="Deposit Type", value="No Deposit")
            customer_type = gr.Dropdown(CUSTOMER_TYPES, label="Customer Type", value="Transient")
            market_segment = gr.Dropdown(MARKET_SEGMENTS, label="Market Segment", value="Online TA")

    predict_btn = gr.Button("Predict", variant="primary")
    output = gr.Textbox(label="Result", lines=3)

    predict_btn.click(
        fn=predict_cancellation,
        inputs=[
            lead_time, stays_in_weekend_nights, stays_in_week_nights, adults,
            booking_changes, adr, total_of_special_requests,
            required_car_parking_spaces, previous_bookings_not_canceled,
            deposit_type, customer_type, market_segment,
        ],
        outputs=output,
    )

if __name__ == "__main__":
    demo.launch()
