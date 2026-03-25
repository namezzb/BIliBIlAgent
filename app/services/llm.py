from openai import OpenAI
from langsmith.wrappers import wrap_openai

from app.services.runtime_audit import LangSmithRuntimeAudit


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
        runtime_audit: LangSmithRuntimeAudit,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.summary_model = summary_model or model
        self.embedding_model = embedding_model
        self.system_prompt = system_prompt
        self.runtime_audit = runtime_audit
        self._client: OpenAI | None = None

    def _get_client(self) -> OpenAI:
        if self._client is None:
            client_kwargs: dict[str, str] = {}
            if self.api_key:
                client_kwargs["api_key"] = self.api_key
            if self.base_url:
                client_kwargs["base_url"] = self.base_url
            self._client = wrap_openai(OpenAI(**client_kwargs))
        return self._client

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        extra_system_messages: list[str] | None = None,
    ) -> str:
        with self.runtime_audit.trace_span(
            name="llm.chat",
            run_type="chain",
            inputs={
                "message_count": len(messages),
                "messages": messages,
                "extra_system_messages": extra_system_messages or [],
                "model": self.model,
            },
            tags=["llm", "chat"],
        ) as trace_run:
            if not self.api_key:
                response = self._fallback_chat(messages)
                trace_run.end(outputs={"response": response, "provider": "fallback"})
                return response

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
                response = f"LLM request failed: {exc}"
                trace_run.end(outputs={"response": response, "provider": "openai"})
                return response

            if isinstance(content, str) and content.strip():
                trace_run.end(outputs={"response": content, "provider": "openai"})
                return content

            response = "The model returned an empty response."
            trace_run.end(outputs={"response": response, "provider": "openai"})
            return response

    def summarize_conversation(self, messages: list[dict[str, str]]) -> str | None:
        with self.runtime_audit.trace_span(
            name="llm.summarize_conversation",
            run_type="chain",
            inputs={
                "message_count": len(messages),
                "messages": messages,
                "model": self.summary_model,
            },
            tags=["llm", "summary"],
        ) as trace_run:
            if not messages:
                trace_run.end(outputs={"summary": None})
                return None
            if not self.api_key:
                trace_run.end(outputs={"summary": None, "reason": "missing_api_key"})
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
                trace_run.end(outputs={"summary": None, "reason": "request_failed"})
                return None

            if isinstance(content, str) and content.strip():
                summary = content.strip()
                trace_run.end(outputs={"summary": summary})
                return summary

            trace_run.end(outputs={"summary": None, "reason": "empty_response"})
            return None

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        with self.runtime_audit.trace_span(
            name="llm.embed_texts",
            run_type="chain",
            inputs={
                "text_count": len(texts),
                "texts": texts,
                "model": self.embedding_model,
            },
            tags=["llm", "embedding"],
        ) as trace_run:
            if not texts:
                trace_run.end(outputs={"embedding_count": 0})
                return []
            if not self.api_key:
                raise RuntimeError("LLM_API_KEY or OPENROUTER_API_KEY is required for embeddings.")

            response = self._get_client().embeddings.create(
                model=self.embedding_model,
                input=texts,
            )
            embeddings = [list(item.embedding) for item in response.data]
            trace_run.end(outputs={"embedding_count": len(embeddings)})
            return embeddings

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
