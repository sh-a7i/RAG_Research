
import streamlit as st
import uuid
from datetime import datetime, timezone

from ragflow_client import ask_ragflow
from experiment_logger import save_response, save_error


# ----------------------------------------
# PAGE CONFIGURATION
# ----------------------------------------

st.set_page_config(
    page_title="RAG Research",
    layout="wide"
)

st.title("RAG Research Experiment")

st.caption(
    "An empirical investigation of retrieval preferences "
    "and answer styles in document question answering."
)


# ----------------------------------------
# LOAD CONFIGURATION
# ----------------------------------------

BASE_URL = st.secrets["ragflow"]["base_url"]
API_KEY = st.secrets["ragflow"]["auth"]["api_key"]


# Each combination maps to one preconfigured RAGFlow assistant.
ASSISTANTS = {
    ("Keyword-oriented", "Direct"): {
        "config_key": "keyword_direct",
        "keyword_weight": 1.0,
        "vector_weight": 0.0,
        "condition": "keyword_direct",
        "prompt_version": "direct_v1"
    },

    ("Keyword-oriented", "Exploratory"): {
        "config_key": "keyword_exploratory",
        "keyword_weight": 1.0,
        "vector_weight": 0.0,
        "condition": "keyword_exploratory",
        "prompt_version": "exploratory_v1"
    },

    ("Semantic-oriented", "Direct"): {
        "config_key": "semantic_direct",
        "keyword_weight": 0.0,
        "vector_weight": 1.0,
        "condition": "semantic_direct",
        "prompt_version": "direct_v1"
    },

    ("Semantic-oriented", "Exploratory"): {
        "config_key": "semantic_exploratory",
        "keyword_weight": 0.0,
        "vector_weight": 1.0,
        "condition": "semantic_exploratory",
        "prompt_version": "exploratory_v1"
    }
}


# ----------------------------------------
# EXPERIMENT INTERFACE
# ----------------------------------------

st.subheader("Experiment configuration")

with st.form("research_form"):

    question_id = st.text_input(
        "Question ID",
        placeholder="Example: Q001"
    )

    question = st.text_area(
        "Enter your question",
        height=120,
        placeholder="Ask a question about the uploaded documents..."
    )

    st.markdown("### Independent Variable 1: Search Focus")

    search_focus = st.select_slider(
        "Select retrieval preference",
        options=[
            "Keyword-oriented",
            "Semantic-oriented"
        ],
        value="Keyword-oriented"
    )

    st.markdown("### Independent Variable 2: Answer Style")

    answer_style = st.select_slider(
        "Select answer style",
        options=[
            "Direct",
            "Exploratory"
        ],
        value="Direct"
    )

    st.caption(
        "Your selections determine which of the four "
        "preconfigured RAGFlow assistants processes the question."
    )

    run_number = st.number_input(
        "Run number",
        min_value=1,
        value=1,
        step=1
    )

    submitted = st.form_submit_button(
        "Submit Question",
        type="primary"
    )


# ----------------------------------------
# PROCESS REQUEST
# ----------------------------------------

if submitted:

    if not question.strip():
        st.warning("Please enter a question.")

    elif not question_id.strip():
        st.warning("Please enter a Question ID.")

    else:

        # Identify the 2 x 2 experimental condition.
        condition = ASSISTANTS[
            (search_focus, answer_style)
        ]

        config_key = condition["config_key"]

        assistant_config = st.secrets[
            "assistants"
        ][config_key]

        chat_id = assistant_config["chat_id"]

        run_id = str(uuid.uuid4())

        timestamp = datetime.now(
            timezone.utc
        ).isoformat()

        # ----------------------------------------
        # EXPERIMENT METADATA
        # ----------------------------------------

        experiment_record = {
            "run_id": run_id,
            "timestamp_utc": timestamp,

            "question_id": question_id.strip(),
            "question": question,

            # Independent variable 1
            "search_focus": search_focus,
            "keyword_weight": condition["keyword_weight"],
            "vector_weight": condition["vector_weight"],

            # Independent variable 2
            "answer_style": answer_style,
            "answer_style_prompt_version": condition["prompt_version"],

            # Combined experimental condition
            "experimental_condition": condition["condition"],

            # RAGFlow configuration
            "ragflow_assistant": config_key,
            "chat_id": chat_id,

            # These must match the saved settings
            # of the four RAGFlow assistants.
            "similarity_threshold": st.secrets["experiment"]["similarity_threshold"],
            "top_n": st.secrets["experiment"]["top_n"],

            "run_number": int(run_number)
        }

        try:

            with st.spinner(
                "Retrieving documents and generating answer..."
            ):

                result = ask_ragflow(
                    question=question,
                    chat_id=chat_id,
                    api_key=API_KEY,
                    base_url=BASE_URL
                )

            # ----------------------------------------
            # SAVE SUCCESSFUL RESPONSE
            # ----------------------------------------

            experiment_record.update({
                "status": "success",
                "answer": result["answer"],
                "references": result["references"],
                "latency_seconds": result["latency_seconds"],
                "usage": result["usage"],
                "raw_response": result["raw_response"]
            })

            save_response(experiment_record)

            # ----------------------------------------
            # DISPLAY ANSWER
            # ----------------------------------------

            st.divider()

            st.subheader("Generated Answer")
            st.write(result["answer"])

            st.subheader("Experiment details")

            col1, col2, col3, col4 = st.columns(4)

            col1.metric(
                "Response time",
                f"{result['latency_seconds']:.2f} sec"
            )

            col2.metric(
                "Search Focus",
                search_focus
            )

            col3.metric(
                "Answer Style",
                answer_style
            )

            col4.metric(
                "Experimental Condition",
                condition["condition"]
            )

            st.success(
                "Response saved successfully to "
                "experiment_data/responses.json"
            )

            with st.expander("Retrieval configuration"):
                st.json({
                    "keyword_weight": condition["keyword_weight"],
                    "vector_weight": condition["vector_weight"],
                    "similarity_threshold": experiment_record["similarity_threshold"],
                    "top_n": experiment_record["top_n"],
                    "assistant": config_key
                })

            with st.expander("Retrieved references"):
                st.json(result["references"])

            with st.expander("Token usage"):
                st.json(result["usage"])

            with st.expander("Full API response"):
                st.json(result["raw_response"])

        except Exception as error:

            error_record = {
                **experiment_record,
                "status": "error",
                "error": str(error),
                "http_response": (
                    error.response.text
                    if getattr(error, "response", None) is not None
                    else None
                )
            }

            save_error(error_record)

            st.error(
                f"RAGFlow request failed: {error}"
            )

            st.info(
                "The error details have been saved to "
                "experiment_data/errors.json."
            )


# ----------------------------------------
# EXPERIMENT DATA LOCATION
# ----------------------------------------

st.divider()

st.caption(
    "Successful responses: experiment_data/responses.json | "
    "Failed requests: experiment_data/errors.json"
)
