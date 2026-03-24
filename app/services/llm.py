from openai import OpenAI


class OpenAICompatibleLLM:
    def __init__(
        self,
        *,
        api_key: str | None,
        base_url: str | None,
        model: str,
        system_prompt: str,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.system_prompt = system_prompt
        self._client: OpenAI | None = None

    def _get_client(self) -> OpenAI:
        if self._client is None:
            client_kwargs: dict[str, str] = {}
            if self.api_key:
                client_kwargs["api_key"] = self.api_key
            if self.base_url:
                client_kwargs["base_url"] = self.base_url
            self._client = OpenAI(**client_kwargs)
        return self._client

    def chat(self, messages: list[dict[str, str]]) -> str:
        if not self.api_key:
            return self._fallback_chat(messages)

        request_messages = [{"role": "system", "content": self.system_prompt}, *messages]

        try:
            completion = self._get_client().chat.completions.create(
                model=self.model,
                messages=request_messages,
                temperature=0.2,
            )
            content = completion.choices[0].message.content
        except Exception as exc:
            return f"LLM request failed: {exc}"

        if isinstance(content, str) and content.strip():
            return content

        return "The model returned an empty response."

    def _fallback_chat(self, messages: list[dict[str, str]]) -> str:
        user_messages = [message["content"] for message in messages if message["role"] == "user"]
        latest_user_message = user_messages[-1].lower() if user_messages else ""

        if len(user_messages) >= 2 and latest_user_message in {
            "我刚刚问了什么？",
            "我刚刚问了什么?",
            "我刚才问了什么？",
            "我刚才问了什么?",
            "what did i just ask?",
            "what was my previous question?",
        }:
            return f"Your previous question was: {user_messages[-2]}"

        return (
            "General chat is available, but no LLM provider is configured yet. "
            "Set LLM_API_KEY to enable model responses."
        )
