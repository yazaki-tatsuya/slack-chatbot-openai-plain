from slack_sdk import WebClient
import conversation_util
import env
import re
import time
# OpenAI
import openai
import pkg_resources
openai.api_key = env.get_env_variable('OPEN_AI_KEY')
model=env.get_env_variable('MODEL')
openai_version = pkg_resources.get_distribution("openai").version
# langchain
from langchain import schema
from langchain.chat_models import ChatOpenAI
# from langchain.chat_models import AzureChatOpenAI
from langchain.schema import HumanMessage

# ロギング
import traceback

def respond_to_message(body, client: WebClient):
    try:
        #-----------------------------------
        # Slackのイベント情報から各種パラメータを取得
        bot_user_id = env.get_env_variable('BOT_USER_ID')
        ts = body["event"]["ts"]
        thread_ts = body["event"].get("thread_ts", None)
        channel = body["event"]["channel"]
        user = body["event"]["user"]
        input_text = schema.HumanMessage(content=body["event"].get("text", None))
        attachment_files = body["event"].get("files", None)
        system_message = schema.SystemMessage(content=env.get_env_variable('SYSTEM_MESSAGE'))

        # やり取りに関するインスタンス生成
        conversation_info = conversation_util.ConversationInfoSlack(
            client=client,
            bot_user_id=bot_user_id,
            ts=ts,
            thread_ts=thread_ts,
            channel=channel,
            user=user,
            human_message_latest=input_text,
            messages=None,
            system_message=system_message
        )
        # メッセージ情報の構築
        conversation_info.build_messages()

        print("==============="+str(conversation_info._messages))
        # OpenAIからの返答を生成
        
        output_text = generate_response_v2(str(conversation_info._messages))
        time.sleep(1)  # n秒待機 (実施しないと「The server responded with: {'ok': False, 'error': 'no_text'}」になる)
        # # Slackに返答
        client.chat_postMessage(channel=channel, text=output_text ,thread_ts=ts)

    except Exception as e:
        print("-------------------------------------------------")
        print("======== react_to_msg 例外発生："+str(e))
        traceback.print_exc()

def generate_response_v2(prompt) ->str:

    print("============ generate_response : TOTAL_PROMPT（過去分含む）："+prompt)
    
    # 言語モデル（OpenAIのチャットモデル）のラッパークラスをインスタンス化
    llm = ChatOpenAI(
        model = "gpt-3.5-turbo",
        openai_api_key=env.get_env_variable('OPEN_AI_KEY'),
        max_tokens=500,
        temperature=0.5
    )

    # APIを使用して、応答を生成します
    #   モデルにPrompt（入力）を与えCompletion（出力）を取得する
    #   SystemMessage: OpenAIに事前に連携したい情報。キャラ設定や前提知識など。
    #   HumanMessage: OpenAIに聞きたい質問
    response = llm(messages=[HumanMessage(content=prompt)])
    
    
    # 文字列から必要なSystemMessageのみを抽出（以下例で言うaaaaaの部分のみ）
    # [SystemMessage(content='aaaaa', additional_kwargs={}), HumanMessage(content='bbbbb', additional_kwargs={}, example=False)]
    pattern = r"SystemMessage\(content='([^']*)'"
    matches = re.findall(pattern, response.content)
    content = ""
    if matches:
        content = matches[0]
    else:
        content = response.content
    print("============ generate_response : COMPLETION："+str(response.content))
    return content