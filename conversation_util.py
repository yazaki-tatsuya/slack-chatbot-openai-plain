import re
from langchain import schema
from slack_sdk import WebClient

class ConversationInfo:
    """
    親クラス
    入力データ（会話履歴＋System Prompt+Human Prompt）をクラス
    開発者は当クラスを継承し、状況に応じたInput情報生成を行う。
    """
    def __init__(
            self, 
            messages:list[schema.BaseMessage]=None, 
            **kwargs) -> None:
        
        # 「_変数」は非公開orプライベート変数
        self._messages = messages
    
    # 以降、@property(Decorator)で、ゲッター（getter）メソッドを定義。
    # 以降、@messages.setter(Decorator)で、セッター（setter）メソッドを定義。

    @property
    def messages(self) -> list[schema.BaseMessage]:
        """
        メッセージ履歴のgetter
        Returns:
            list: メッセージ履歴のリスト
        """
        return self._messages
    
    @messages.setter
    def messages(self, value:list[schema.BaseMessage]):
        """
        メッセージ履歴のsetter
        Args:
            value (list[schema.BaseMessage]): メッセージ履歴
        """
        self._messages = value
  
    def build_messages(self, **kwargs):
        """
        messagesを構築するための処理を実装するメソッド。
        必須でなく、メンバ変数「messages」を直接セットする事も可能
        """
        pass

def remove_mention_str(remove_user_id:str, target_text:str) -> str:
    """
    文頭/文末のボットへのメンション文字列"<@Uxxxx>"を除外する

    Args:
        remove_user_id (str): 除外したいボットのユーザーID
        target_text (str): 除外を行いたいテキスト文字列

    Returns:
        str: メンション文字列が除外された新しいテキスト文字列
    """
    # 例：^<@UAAAAAAAAAA.*?>|<@UAAAAAAAAAA.*?>$
    pattern = f'^<@{remove_user_id}.*?>|<@{remove_user_id}.*?>$'
    # target_textのうち、パターンにマッチした部分を除去
    ret = re.sub(pattern, '', target_text.strip())
    return ret


def get_thread_history(
            client:WebClient, 
            bot_user_id:str, 
            ts: str, 
            thread_ts:str,
            channel:str) ->list:
    """
    スレッドメッセージの履歴も含めた、一連のやり取りの履歴を取得する。
    （親メッセージ＋conversation_repliesのlimitの件数まで取得）
        
    Args:
        bot_user_id (str): ボットのユーザーID。メッセージが"ai" or "human"を判定するために必要。

        client (WebClient)          :Slackとインタラクトする為のWebClientクラスのインスタンス
        bot_user_id (str)           :ボットのユーザーID
        ts (str)                    :ユーザーメッセージのタイムスタンプ
        thread_ts (str)             :ユーザーメッセージがスレッドの一部である場合の親メッセージのtimestamp
        channel (str)               :チャンネルID

    Returns:
        list: schema.HumanMessage または schema.SystemMessageが格納されたlist

    """
    history = []
    # 会話履歴を取得
    resp = client.conversations_replies(
        channel=channel, inclusive=False, latest=ts, ts=thread_ts)
    messages = resp.get("messages")

    # ボットの応答はAIMessageに、人からの回答はHumanMessageに格納
    for message in messages:
        if message.get("user", "-") == bot_user_id:
            content=message.get("text", "**** メッセージの取得に失敗しました ****")
            history.append(schema.AIMessage(content=content))
        else:
            content=message.get("text", "**** メッセージの取得に失敗しました ****")
            history.append(schema.HumanMessage(content=content))
    return history

class ConversationInfoSlack(ConversationInfo):
    """
    TaskInputChatを継承した子クラス
    Slackからのメッセージ(処理依頼)に対して、メッセージ情報を保持するためのクラス
    """
    def __init__(self, 
                client:WebClient, 
                bot_user_id:str, 
                ts:str,
                thread_ts:str,
                channel:str, 
                user:str,
                human_message_latest:schema.HumanMessage=None,
                messages:list[schema.BaseMessage]=None,
                system_message:schema.SystemMessage=None,
                **kwargs) -> None:
        """
        コンストラクタ。
        Args:
            client (WebClient)          :Slackとインタラクトする為のWebClientクラスのインスタンス
            bot_user_id (str)           :ボットのユーザーID
            ts (str)                    :ユーザーメッセージのタイムスタンプ
            thread_ts (str)             :ユーザーメッセージがスレッドの一部である場合の親メッセージのtimestamp
            channel (str)               :チャンネルID
            user (str)                  :投稿したユーザーのSlackユーザーID
            human_message_latest (schema.HumanMessage)  : 最新のユーザーメッセージ (省略可能)
            messages (list[schema.BaseMessage])
                                        :メッセージ履歴 (省略された場合は、後からsetterから格納するか、build_messages()から構築)
                                         schema.BaseMessageクラスのインスタンスのlistを受け取ることができます(Noneのため省略可能)
            system_message (schema.SystemMessage)       : システムメッセージ (省略可能)
        Notes:
            - system_message、human_message_latestを設定する事で、build_messages()を呼び、過去スレッド履歴も含めたmessagesが構築できる。
            - build_messages()を使用せずに、直接メンバ変数messagesに値を格納する事も可能。
        """

        # 親クラスのコンストラクタを呼ぶ
        super().__init__(messages=messages,**kwargs)

        self.client = client
        self.bot_user_id = bot_user_id
        self.ts = ts
        self.thread_ts = thread_ts
        self.channel = channel
        self.user = user
        self.human_message_latest = human_message_latest
        self.system_message = system_message
        
    def build_messages(self, **kwargs):
        """
        messagesを構築するための処理を実装するメソッド。
        必須でなく、メンバ変数「messages」を直接セットする事も可能
        """
        ret = []

        # システムメッセージを追加
        if not self.system_message is None:
            ret.append(self.system_message)

        # スレッド履歴を追加
        if self.thread_ts is not None:        
            thread_history = get_thread_history(
                client=self.client,
                bot_user_id=self.bot_user_id,
                ts=self.ts,
                thread_ts=self.thread_ts,
                channel=self.channel
            )
            ret.extend(thread_history)

        # 最新のメッセージを追加
        ret.append(self.human_message_latest)

        # ボットへの文頭/文末のメンションを除去する
        # Before例: <@U03U3JE514N>\n私は赤色の携帯が欲しいです。
        # After例: '私は赤色の携帯が欲しいです。'
        for message in ret:
            message.content = remove_mention_str(self.bot_user_id, message.content)
        
        self._messages = ret