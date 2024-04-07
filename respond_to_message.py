from slack_sdk import WebClient
import conversation_util
from datetime import datetime, timedelta
import env
import json
import re
import time
# OpenAI
import openai
import tiktoken
from tiktoken.core import Encoding
import pkg_resources
openai.api_key = env.get_env_variable('OPEN_AI_KEY')
model=env.get_env_variable('MODEL')
openai_version = pkg_resources.get_distribution("openai").version
# langchain
from langchain import schema
from langchain_community.chat_models import ChatOpenAI
# from langchain.chat_models import AzureChatOpenAI
# from langchain.schema import HumanMessage
from langchain.schema.messages import HumanMessage
from langchain_core.messages.ai import AIMessage

# ロギング
import traceback
import logging

# 設定とロギングの初期化
OPEN_AI_KEY = env.get_env_variable('OPEN_AI_KEY')
MODEL = env.get_env_variable('OPEN_AI_MODEL')
BOT_USER_ID = env.get_env_variable('BOT_USER_ID')
SYSTEM_MESSAGE = env.get_env_variable('SYSTEM_MESSAGE')

class SlackBotHandler:
    def __init__(self, slack_client: WebClient):
        self.slack_client = slack_client
        self.chat_open_ai = ChatOpenAI(
            model=MODEL,
            openai_api_key=OPEN_AI_KEY,
            max_tokens=500,
            temperature=0.5
        )

    def send_loading_message(self, channel, ts):
        loading_message_view = {
            "blocks": [
                {
                    "type": "context",
                    "elements": [
                        {
                            "type": "mrkdwn",
                            "text": f"{MODEL}で処理中... :now-loading:"
                        }
                    ]
                }
            ]
        }
        resp = self.slack_client.chat_postMessage(channel=channel, thread_ts=ts, blocks=json.dumps(loading_message_view["blocks"]))
        return resp["ts"]

def respond_to_message(body, client: WebClient,logger:logging.Logger):
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

        #-----------------------------------
        # Loadingメッセージを通知
        #-----------------------------------
        handler = SlackBotHandler(client)
        loading_message_ts = handler.send_loading_message(channel, ts)

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

        logger.info(f"respond_to_message - メッセージのビルド完了： {str(conversation_info._messages)}")
        
        # Function Callingによる呼び出し関数の判断
        execute_function_info = generate_response_function_calling(conversation_info._messages,get_function_descriptions())
        logger.info(f"respond_to_message - 使用する関数の判定が完了： {str(execute_function_info)}")
        # 判定結果から実行対象の関数名を取得する
        if 'function_call' in execute_function_info.additional_kwargs:
            chosen_function = execute_function_info.additional_kwargs['function_call']['name']
        else:
            # function_call キーが存在しない場合のエラーハンドリング
            chosen_function = "chat_with_gpt"

        # (1) 呼び出し関数が「chat_with_gpt」の場合
        if chosen_function=="chat_with_gpt":
            # OpenAIからの返答を生成
            output_text = generate_response_v2(conversation_info._messages)
            # output_text = generate_response_v2(str(conversation_info._messages))
            time.sleep(1)  # n秒待機 (実施しないと「The server responded with: {'ok': False, 'error': 'no_text'}」になる)
            logger.info(f"respond_to_message - (1)chat_with_gptの応答予定内容： {str(output_text)}")

            # Slackに応答
            client.chat_postMessage(channel=channel, text=output_text ,thread_ts=ts)
        
        # (2) 呼び出し関数が「chat_with_gpt」以下の場合 (「get_flight_info」の場合)
        else:
            # 判定結果から関数の引数を取得
            arguments = json.loads(execute_function_info.additional_kwargs['function_call']['arguments'])
            origin = arguments.get("departure")
            destination = arguments.get("destination")

            # 該当の関数を呼び出し
            chosen_function = eval(execute_function_info.additional_kwargs['function_call']['name'])
            flight = chosen_function(origin, destination)
            logger.info(f"respond_to_message - (2)get_flight_infoの応答予定内容： {str(flight)}")
        
            # Slackに応答
            client.chat_postMessage(channel=channel, text=str(flight) ,thread_ts=ts)

    except Exception as e:
        logger.info(f"respond_to_message - 例外発生： {str(e)}")
        traceback.print_exc()

    finally:
        # Loadingメッセージを削除
        client.chat_delete(channel=channel, ts=loading_message_ts)

def generate_response_v2(prompt) ->str:
    print("============ generate_response : TOTAL_PROMPT（過去分含む）："+str(prompt))
    
    # 言語モデル（OpenAIのチャットモデル）のラッパークラスをインスタンス化
    llm = ChatOpenAI(
        model = "gpt-3.5-turbo",
        openai_api_key=env.get_env_variable('OPEN_AI_KEY'),
        max_tokens=500,
        temperature=0.5
    )

    # LLMモデル実行クラスのインスタンス化
    llm_exec = LlmModelExecuter(llm,prompt)
    response = llm_exec.execute_llm_model()
    
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

class LlmModelExecuter:
    """
    言語モデルを実行するクラス
    """
    def __init__(self, 
                 llm_chat:ChatOpenAI, 
                 prompt:list[schema.BaseMessage]) -> None:
        self.llm_chat = llm_chat
        self.prompt = prompt

    def execute_llm_model(self) ->schema.BaseMessage:

        # 元のメッセージリストから、contentのみを抽出し、新しいリストに格納
        content_str = ""
        for message in self.prompt:
            print(message)
            content_str += message.content
        encoding: Encoding = tiktoken.encoding_for_model("gpt-4")
        tokens = encoding.encode(content_str)
        tokens_count = len(tokens)
        print("================================= TOKEN_NUM = "+str(tokens_count))

        # APIを使用して、応答を生成します
        #   モデルにPrompt（入力）を与えCompletion（出力）を取得する
        #   SystemMessage: OpenAIに事前に連携したい情報。キャラ設定や前提知識など。
        #   HumanMessage: OpenAIに聞きたい質問
        response = self.llm_chat(messages=[HumanMessage(content=str(self.prompt))])

        return response

def get_function_descriptions():

    # GPTに渡す関数の説明
    function_descriptions = [
        {
            "name": "get_flight_info",
            "description": "出発地と目的地の2つの情報からフライト情報を取得する",
            "parameters": {
                "type": "object",
                "properties": {
                    "departure": {
                        "type": "string",
                        "description": "出発地の空港。(例) HND",
                    },
                    "destination": {
                        "type": "string",
                        "description": "目的地の空港。(例) CDG",
                    },
                },
                "required": ["departure", "destination"],
            },
        },
        {
            "name": "chat_with_gpt",
            "description": "フライト情報取得「以外」のものは、全てこちらを使う",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "ユーザーからの質問クエリ",
                    },
                },
                "required": ["prompt"],
            },
        },
    ]
    return function_descriptions

def generate_response_function_calling(user_prompt,function_descriptions) ->AIMessage:
    print("============ function_calling : TOTAL_PROMPT（過去分含む）："+str(user_prompt))

    # llmモデルのインスタンス生成
    llm = ChatOpenAI(
        model = "gpt-3.5-turbo",
        openai_api_key=env.get_env_variable('OPEN_AI_KEY'),
        max_tokens=500,
        temperature=0.5
    )
    # 関数の使用判定(1個目)
    first_response = llm.predict_messages(
        [HumanMessage(content=str(user_prompt))],
        functions=function_descriptions
    )
    print(type(first_response))
    return first_response

    # # 判定結果から関数の引数を取得
    # arguments = json.loads(first_response.additional_kwargs['function_call']['arguments'])
    # origin = arguments.get("departure")
    # destination = arguments.get("destination")

    # # 取得した引数を与えて、関数を呼び出し
    # chosen_function = eval(first_response.additional_kwargs['function_call']['name'])
    # flight = chosen_function(origin, destination)

    # print("============ function_calling : COMPLETION："+str(flight))
    # return flight

# 出発地と目的地を引数としてフライト情報を取得する関数
def get_flight_info(departure, destination):
    """
    出発地と目的地の間のフライト情報を取得する関数
    """
    # デモのためのダミーのフライト情報（本来はDBやAPI経由で取得する）
    flight_info = {
        "departure": departure,
        "destination": destination,
        "datetime": str(datetime.now() + timedelta(hours=2)),
        "airline": "JAL",
        "flight": "JL0006",
    }
    # フライト情報をJSON形式で返す
    return json.dumps(flight_info)