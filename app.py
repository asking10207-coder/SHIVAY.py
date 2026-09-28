import streamlit as st
import pandas as pd
import numpy as np
import joblib


# ------------------------------------------------
# PAGE CONFIGURATION
# ------------------------------------------------

st.set_page_config(
    page_title="SHIVAY Food Delivery Analytics",
    page_icon="🍔",
    layout="wide"
)


# ------------------------------------------------
# LOAD MODELS
# ------------------------------------------------

linear_model = joblib.load(
    "shivay_linear_model.pkl"
)

logistic_model = joblib.load(
    "shivay_logistic_model.pkl"
)


# ------------------------------------------------
# HEADER
# ------------------------------------------------

st.title(
    "🍔 SHIVAY Food Delivery"
)

st.subheader(
    "Predictive Analytics & Managerial Decision Support Tool"
)

st.write(
    """
    This application uses machine learning to estimate
    delivery time and identify the probability of a
    late food-delivery order.
    """
)

st.divider()


# ------------------------------------------------
# SIDEBAR
# ------------------------------------------------

st.sidebar.header(
    "Navigation"
)

option = st.sidebar.radio(
    "Select an option",
    [
        "Single Order Prediction",
        "Upload Order Dataset",
        "About the Model"
    ]
)


# =================================================
# SINGLE ORDER PREDICTION
# =================================================

if option == "Single Order Prediction":

    st.header(
        "Single Order Prediction"
    )

    col1, col2 = st.columns(2)

    # ---------------------------------------------
    # ORDER VALUE
    # ---------------------------------------------

    with col1:

        order_value = st.number_input(
            "Order Value",
            min_value=0.0,
            value=500.0
        )

        delivery_fee = st.number_input(
            "Delivery Fee",
            min_value=0.0,
            value=40.0
        )

        commission_fee = st.number_input(
            "Commission Fee",
            min_value=0.0,
            value=80.0
        )

        payment_processing_fee = st.number_input(
            "Payment Processing Fee",
            min_value=0.0,
            value=10.0
        )

        refunds = st.number_input(
            "Refunds / Chargebacks",
            min_value=0.0,
            value=0.0
        )

    # ---------------------------------------------
    # CATEGORICAL VARIABLES
    # ---------------------------------------------

    with col2:

        payment_method = st.selectbox(
            "Payment Method",
            [
                "UPI",
                "Credit Card",
                "Debit Card",
                "Cash",
                "Net Banking"
            ]
        )

        discount = st.selectbox(
            "Discounts and Offers",
            [
                "No Discount",
                "10% Off",
                "20% Off",
                "Free Delivery"
            ]
        )

        order_hour = st.slider(
            "Order Hour",
            0,
            23,
            19
        )

        day = st.selectbox(
            "Day of Week",
            [
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
                "Sunday"
            ]
        )

        weekend = 1 if day in [
            "Saturday",
            "Sunday"
        ] else 0


    # ---------------------------------------------
    # CREATE INPUT DATAFRAME
    # ---------------------------------------------

    input_data = pd.DataFrame({

        "Order Value": [order_value],

        "Delivery Fee": [delivery_fee],

        "Payment Method": [payment_method],

        "Discounts and Offers": [discount],

        "Commission Fee": [commission_fee],

        "Payment Processing Fee":
            [payment_processing_fee],

        "Refunds/Chargebacks":
            [refunds],

        "Order_Hour": [order_hour],

        "Day_of_Week": [day],

        "Weekend": [weekend]
    })


    st.divider()


    # ---------------------------------------------
    # PREDICT
    # ---------------------------------------------

    if st.button(
        "🔮 Predict Order Performance",
        use_container_width=True
    ):

        delivery_prediction = (
            linear_model.predict(
                input_data
            )[0]
        )

        late_probability = (
            logistic_model.predict_proba(
                input_data
            )[0][1]
        )


        # -----------------------------------------
        # DISPLAY RESULTS
        # -----------------------------------------

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Predicted Delivery Time",
                f"{delivery_prediction:.1f} min"
            )

        with col2:

            st.metric(
                "Late Delivery Probability",
                f"{late_probability*100:.1f}%"
            )

        with col3:

            if late_probability >= 0.70:

                risk = "HIGH"

            elif late_probability >= 0.40:

                risk = "MEDIUM"

            else:

                risk = "LOW"

            st.metric(
                "Risk Level",
                risk
            )


        # -----------------------------------------
        # MANAGERIAL INTERPRETATION
        # -----------------------------------------

        st.subheader(
            "Managerial Interpretation"
        )

        if late_probability >= 0.70:

            st.error(
                """
                High late-delivery risk.
                
                Managerial action:
                Consider proactive monitoring,
                customer communication and operational
                review.
                """
            )

        elif late_probability >= 0.40:

            st.warning(
                """
                Moderate late-delivery risk.
                
                Managerial action:
                Monitor the order and review operational
                conditions.
                """
            )

        else:

            st.success(
                """
                Low late-delivery risk.
                
                No immediate intervention is indicated
                by the model.
                """
            )


        st.info(
            """
            Note: The prediction is a decision-support
            signal and should be combined with managerial
            judgement.
            """
        )


# =================================================
# UPLOAD DATASET
# =================================================

elif option == "Upload Order Dataset":

    st.header(
        "Upload Order Dataset"
    )

    st.write(
        """
        Upload a CSV containing the variables required
        by the predictive models.
        """
    )

    uploaded_file = st.file_uploader(
        "Upload CSV",
        type=["csv"]
    )


    if uploaded_file is not None:

        data = pd.read_csv(
            uploaded_file
        )

        st.subheader(
            "Uploaded Data"
        )

        st.dataframe(
            data,
            use_container_width=True
        )


        if st.button(
            "Generate Predictions"
        ):

            delivery_predictions = (
                linear_model.predict(data)
            )

            late_probabilities = (
                logistic_model
                .predict_proba(data)[:, 1]
            )


            result = data.copy()

            result[
                "Predicted Delivery Time (Min)"
            ] = np.round(
                delivery_predictions,
                2
            )

            result[
                "Late Delivery Probability (%)"
            ] = np.round(
                late_probabilities * 100,
                2
            )

            result[
                "Risk Level"
            ] = pd.cut(
                late_probabilities,
                bins=[
                    -0.01,
                    0.40,
                    0.70,
                    1.0
                ],
                labels=[
                    "Low",
                    "Medium",
                    "High"
                ]
            )


            st.subheader(
                "Prediction Results"
            )

            st.dataframe(
                result,
                use_container_width=True
            )


            csv = result.to_csv(
                index=False
            ).encode("utf-8")


            st.download_button(
                "Download Predictions",
                csv,
                "SHIVAY_predictions.csv",
                "text/csv"
            )


# =================================================
# ABOUT MODEL
# =================================================

else:

    st.header(
        "About the Predictive Models"
    )

    st.write(
        """
        ### Linear Regression
        
        The linear regression model predicts
        expected delivery time in minutes.
        
        ### Logistic Regression
        
        The logistic regression model estimates
        the probability that an order will take
        more than 75 minutes.
        
        ### Managerial Use
        
        The tool is intended to support managerial
        decisions rather than replace human judgement.
        """
    )

    st.subheader(
        "Model Variables"
    )

    st.write(
        [
            "Order Value",
            "Delivery Fee",
            "Payment Method",
            "Discounts and Offers",
            "Commission Fee",
            "Payment Processing Fee",
            "Refunds / Chargebacks",
            "Order Hour",
            "Day of Week",
            "Weekend"
        ]
    )
