# streamlit 페이지로 front 구성

'''
｢ 관세청 개인통관 안내 GPT  ｣

'''

import streamlit as st

import os

import requests
import json

# Setup session states
def reset_messages():
    st.session_state["messages_op"] = [
        {"role": "assistant", "content": "- 이곳은 해외직구, 여행자통관, 우편통관, 이사화물에 대하여 질문할 수 있는 챗봇입니다.\n- 안내챗봇의 답변은 개인적인 사정을 고려함이 없이 일반적인 사항에 대한 안내임으로 법적인 효력이 없습니다."}
    ]
    
if "messages_op" not in st.session_state:
    reset_messages()     
    
# top_k = st.sidebar.selectbox('참조할 최대 문서 갯수를 선택하세요.',[1,2,3,4,5,6,7,8,9,10], index = 6)
# st.sidebar.write(f"현재 선택된 참조할 최대 문서 갯수는 {top_k}개 입니다.")
top_k = 3



# Main page
def main():
    st.title("관세청 개인통관 안내 on-premise")
    for message in st.session_state["messages_op"]:
        if "hide" not in message:
            with st.chat_message(message["role"]):
                st.markdown(message['content'])

    
    if prompt := st.chat_input("질문/대답을 여기 입력해주세요."):
        st.chat_message("user").markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner(text="생각 중입니다.."):
                diction = {
                    'prompt': prompt,
                    'top_k': top_k,
                }
                
                try:
                    response = requests.post("http://localhost:8609/kullm", data=json.dumps(diction))
                    if response:
                        result = json.loads(response.text)
                    else:
                        result = {}
                        result['doc_lst'] = ''
                        result['answer'] = ''
                    
                    out_msg_lst = result['doc_lst']
                    answer = result['answer']
                    
                    if len(answer) <= 1:
                        answer = '모델이 질문을 이해하지 못했습니다. 다른 표현으로 질문 부탁드리겠습니다.'

                    st.write(answer)
                    st.write("-"*50)
                    referred_documents = "### 참조 문서 목록\n - "+"\n - ".join(out_msg_lst)
                    st.markdown(referred_documents)     
                except requests.exceptions.RequestException as e:
                    out_msg_lst = ''
                    answer = ''
                    
                    referred_documents = "### 참조 문서 목록\n - "+"\n - ".join(out_msg_lst)
                    st.warning(f"현재 모델이 업로드 중입니다. 잠시후 다시 시도해 주세요. 문제가 계속되는 경우 관리자에게 연락바랍니다.")
            
        st.session_state["messages_op"].append({"role": "user", "content": prompt})
        st.session_state["messages_op"].append({"role": "assistant", "content": answer})
        st.session_state["messages_op"].append({"role": "assistant", "content": referred_documents})
        
    st.button("대화 초기화", on_click=reset_messages)

if __name__=="__main__":
    main()
