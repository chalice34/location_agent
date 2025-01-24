import streamlit as st
import boto3
import uuid
from dotenv import load_dotenv,dotenv_values


envvals=dotenv_values()


region_name = "ap-south-1"
bedrock_agent_runtime = boto3.client(service_name='bedrock-agent-runtime', region_name=region_name,aws_access_key_id=st.secrets["aws_access_key_id"],aws_secret_access_key=st.secrets["aws_secret_access_key"])

def invoke_agent(agent_id, agent_alias_id, session_id, prompt):
    """
    Sends a prompt for the agent to process and respond to.

    :param agent_id: The unique identifier of the agent to use.
    :param agent_alias_id: The alias of the agent to use.
    :param session_id: The unique identifier of the session. Use the same value across requests
                        to continue the same conversation.
    :param prompt: The prompt that you want Claude to complete.
    :return: Inference response from the model.
    """
    completion = ""
    try:
        # Note: The execution time depends on the foundation model, complexity of the agent,
        # and the length of the prompt. In some cases, it can take up to a minute or more to
        # generate a response.
        response = bedrock_agent_runtime.invoke_agent(
            agentId=agent_id,
            agentAliasId=agent_alias_id,
            sessionId=session_id,
            inputText=prompt,
        )

        for event in response.get("completion"):
            chunk = event["chunk"]
            completion = completion + chunk["bytes"].decode()

    except Exception as e:
        print(f"Couldn't invoke agent. {e}")

    return completion


st.title("Location Finder")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [{"role":"assistant","content":"write query in format of 'query' and 'location' example:'find hospitals in Visakhapatnam'"}]

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("find hospitals in Visakhapatnam"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Ensure there is a message to process before invoking the agent
    if len(st.session_state.messages) > 0:
        session_id = str(uuid.uuid4())
        pr = st.session_state.messages[-1]
        # print(f"pr: {pr['content']}")
        x = invoke_agent(st.secrets["agent_id"],st.secrets["agent_alias"], session_id, prompt=pr["content"])
        response = f"{x}"

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role":"assistant","avatar":"✈️", "content": response})
