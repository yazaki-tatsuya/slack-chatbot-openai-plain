def get_env_variable(key):

    env_variable_dict = {
        # ------------------------------------
        # Azure Storage
        # ------------------------------------
        "AZURE_STORAGE_NAME" : "",
        "AZURE_STORAGE_KEY" : "",

        # ------------------------------------
        # OpenAI(Azure)
        # ------------------------------------
        # "OPEN_AI_KEY" : "",
        # "OPEN_AI_BASE" : "https://aaaaa.openai.azure.com/",
        # "OPEN_AI_TYPE" : 'azure',
        # "OPEN_AI_VERSION" : '2023-03-15-preview',
        # "OPEN_AI_DEPLOY_NAME" : 'gpt4_32k',
        # "MODEL" : '',
        # "SYSTEM_MESSAGE" : "あなたはキャリアコンサルタントです。クライアントに寄り添い、問いかけを行い、内省を促してください。",

        # ------------------------------------
        # OpenAI(Rainbow)
        # ------------------------------------
        "OPEN_AI_KEY" : "",
        "MODEL" : "",
        "SYSTEM_MESSAGE" : "あなたはキャリアコンサルタントです。クライアントに寄り添い、問いかけを行い、内省を促してください。",

        # ------------------------------------
        # App名：Slack_Python_Flask
        # ------------------------------------
        # BotユーザーID
        "BOT_USER_ID" : "",
        # Botトークン（Flask）
        "WEBAPPS_SLACK_TOKEN" : "",
        "WEBAPPS_SIGNING_SECRET" : "",

        # Botトークン（ソケットモード）
        "SOCK_SLACK_BOT_TOKEN" : "",
        "SOCK_SLACK_APP_TOKEN" : ""
    }
    ret_val = env_variable_dict.get(key, None)
    return ret_val