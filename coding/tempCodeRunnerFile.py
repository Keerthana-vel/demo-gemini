

    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

    # Display chat messages and bot response
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                if handle_greeting(prompt):
                    response = {"output_text": "Hello! Provide a question related to AV receiver!"}
                else:
                    response = user_input(prompt)
                placeholder = st.empty()
                full_response = ''
               # Ensure that response['output_text'] is handled correctly
                output_text = response.get('output_text', "")
            
                if isinstance(output_text, str):