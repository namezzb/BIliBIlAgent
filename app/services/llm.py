from openai import OpenAI


class OpenAICompatibleLLM:
    def __init__(
        self,
        *,
        api_key: str | None,
        base_url: str | None,
        model: str,
        summary_model: str | None,
        embedding_model: str,
        system_prompt: str,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.summary_model = summary_model or model
        self.embedding_model = embedding_model
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

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        extra_system_messages: list[str] | None = None,
    ) -> str:
        if not self.api_key:
            return self._fallback_chat(messages)

        system_messages = [{"role": "system", "content": self.system_prompt}]
        if extra_system_messages:
            system_messages.extend(
                {"role": "system", "content": content}
                for content in extra_system_messages
                if content.strip()
            )

        try:
            content = self._create_chat_completion(
                messages=[*system_messages, *messages],
                model=self.model,
                temperature=0.2,
            )
        except Exception as exc:
            return f"LLM request failed: {exc}"

        if isinstance(content, str) and content.strip():
            return content

        return "The model returned an empty response."

    def summarize_conversation(self, messages: list[dict[str, str]]) -> str | None:
        if not messages:
            return None
        if not self.api_key:
            return None

        transcript = "\n".join(
            f"{message['role'].upper()}: {message['content']}" for message in messages
        )
        summary_prompt = (
            "Summarize the prior conversation for future turns. "
            "Keep only stable facts, unresolved goals, and important constraints. "
            "Do not include roleplay, filler, or formatting. "
            "Return plain text in at most 6 short lines."
        )

        try:
            content = self._create_chat_completion(
                messages=[
                    {"role": "system", "content": summary_prompt},
                    {
                        "role": "user",
                        "content": f"Conversation transcript:\n{transcript}",
                    },
                ],
                model=self.summary_model,
                temperature=0.1,
            )
        except Exception:
            return None

        if isinstance(content, str) and content.strip():
            return content.strip()
        return None

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if not self.api_key:
            raise RuntimeError("LLM_API_KEY or OPENROUTER_API_KEY is required for embeddings.")

        response = self._get_client().embeddings.create(
            model=self.embedding_model,
            input=texts,
        )
        return [list(item.embedding) for item in response.data]

    def _create_chat_completion(
        self,
        *,
        messages: list[dict[str, str]],
        model: str,
        temperature: float,
    ) -> str | None:
        completion = self._get_client().chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
        )
        return completion.choices[0].message.content

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
