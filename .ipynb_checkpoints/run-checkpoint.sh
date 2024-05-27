# python3 server.py를 실행합니다.
nohup python3 server.py 1> /dev/null 2>&1 &

# streamlit을 실행하고 출력을 /dev/null로 리디렉션합니다.
nohup streamlit run app.py --server.fileWatcherType none --server.port 18000 1> /dev/null 2>&1 &