# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Base LLM class definition."""

import traceback
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from typing_extensions import Unpack

from graphrag.llm.types import (
    LLM,
    ErrorHandlerFn,
    LLMInput,
    LLMOutput,
)

TIn = TypeVar("TIn")
TOut = TypeVar("TOut")
import tiktoken
enc = tiktoken.get_encoding("o200k_base")
import os 

class BaseLLM(ABC, LLM[TIn, TOut], Generic[TIn, TOut]):
    """LLM Implementation class definition."""

    _on_error: ErrorHandlerFn | None

    def on_error(self, on_error: ErrorHandlerFn | None) -> None:
        """Set the error handler function."""
        self._on_error = on_error

    @abstractmethod
    async def _execute_llm(
        self,
        input: TIn,
        **kwargs: Unpack[LLMInput],
    ) -> TOut | None:
        pass

    async def __call__(
        self,
        input: TIn,
        **kwargs: Unpack[LLMInput],
    ) -> LLMOutput[TOut]:
        """Invoke the LLM."""
        is_json = kwargs.get("json") or False
        output = ""
        if is_json:
            output = await self._invoke_json(input, **kwargs)
        else:
            output = await self._invoke(input, **kwargs)
        open(os.environ['output_dir'],"a").write("<START>\n"+str(input)+"\n<OUTPUT>\n"+str(output)+"\n<END>\n\n")
        if not "token_count" in os.environ:
            os.environ['token_count'] = "0"
        os.environ['token_count'] = str(int(os.environ['token_count'])+len(enc.encode(input)))
        return output

    async def _invoke(self, input: TIn, **kwargs: Unpack[LLMInput]) -> LLMOutput[TOut]:
        try:
            output = await self._execute_llm(input, **kwargs)
            return LLMOutput(output=output)
        except Exception as e:
            stack_trace = traceback.format_exc()
            if self._on_error:
                self._on_error(e, stack_trace, {"input": input})
            raise

    async def _invoke_json(
        self, input: TIn, **kwargs: Unpack[LLMInput]
    ) -> LLMOutput[TOut]:
        msg = "JSON output not supported by this LLM"
        raise NotImplementedError(msg)
