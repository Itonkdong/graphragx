"""Shared selection service for validated programmatic or interactive choices."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from pipeline.services import AbstractService

SelectionOption = TypeVar("SelectionOption")


class SelectionService(AbstractService):
    """Resolve selections from constructor values or interactive prompts."""

    def __init__(
        self,
        input_func: Callable[[str], str] | None = None,
        output_func: Callable[[str], None] | None = None,
    ):
        self.input_func = input_func or input
        self.output_func = output_func or print

    def resolve_choice(
        self,
        provided_value: str | None,
        options: dict[str, SelectionOption],
        prompt_title: str,
        prompt_help: str,
        recommended_id: str,
        invalid_exception_type: type[Exception],
        value_getter: Callable[[SelectionOption], str],
        label_getter: Callable[[SelectionOption], str],
    ) -> str:
        """Return a validated constructor value or interactively prompt for one."""
        if provided_value is not None:
            if provided_value not in options:
                raise invalid_exception_type(f"Invalid selection: {provided_value}")
            return provided_value

        return self.prompt_for_choice(
            options=options,
            prompt_title=prompt_title,
            prompt_help=prompt_help,
            recommended_id=recommended_id,
            value_getter=value_getter,
            label_getter=label_getter,
            invalid_exception_type=invalid_exception_type,
        )

    def prompt_for_choice(
        self,
        options: dict[str, SelectionOption],
        prompt_title: str,
        prompt_help: str,
        recommended_id: str,
        value_getter: Callable[[SelectionOption], str],
        label_getter: Callable[[SelectionOption], str],
        invalid_exception_type: type[Exception],
    ) -> str:
        """Prompt interactively until a valid numbered choice is selected."""
        option_items = list(options.values())
        while True:
            self.output_func(f"\n{prompt_title}")
            self.output_func(prompt_help)
            for index, option in enumerate(option_items, start=1):
                option_value = value_getter(option)
                recommended_suffix = " (Recommended)" if option_value == recommended_id else ""
                self.output_func(f"{index}) {label_getter(option)}{recommended_suffix}")

            try:
                raw_value = self.input_func("Enter the number of your choice: ").strip()
            except (EOFError, KeyboardInterrupt) as error:
                raise invalid_exception_type(
                    f"Unable to read interactive input for {prompt_title}."
                ) from error

            if not raw_value.isdigit():
                self.output_func("Invalid selection. Please enter a valid number.")
                continue

            selected_index = int(raw_value)
            if selected_index < 1 or selected_index > len(option_items):
                self.output_func("Invalid selection. Please choose one of the listed options.")
                continue

            selected_option = option_items[selected_index - 1]
            return value_getter(selected_option)

    def resolve_text(
        self,
        provided_value: str | None,
        prompt: str,
        invalid_exception_type: type[Exception],
    ) -> str:
        """Return a non-empty constructor value or prompt until one is entered."""
        if provided_value is not None:
            value = provided_value.strip()
            if not value:
                raise invalid_exception_type("The value cannot be empty.")
            return value
        while True:
            try:
                value = self.input_func(prompt).strip()
            except (EOFError, KeyboardInterrupt) as error:
                raise invalid_exception_type("Unable to read interactive input.") from error
            if value:
                return value
            self.output_func("Invalid value. Please enter a non-empty model name.")
