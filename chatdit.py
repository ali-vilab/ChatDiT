import openai
import torch

from agents import (
    InstructionParsingAgent,
    StrategyPlanningAgent,
    ExecutionAgent,
    MarkdownAgent
)

__all__ = ['ChatDiT']


class ChatDiT:

    def __init__(self, client=openai.OpenAI(), device=torch.device('cuda:0')):
        self.instruction_parsing_agent = InstructionParsingAgent(client=client)
        self.strategy_planning_agent = StrategyPlanningAgent(client=client)
        self.execution_agent = ExecutionAgent(device=device)
        self.markdown_agent = MarkdownAgent(client=client)
    
    def chat(self, message, images=[], return_markdown=False):
        instruction_parsing_output_json = self.instruction_parsing_agent(
            instruction=message,
            images=images
        )
        strategy_planning_output_json = self.strategy_planning_agent(
            instruction_parsing_output_json=instruction_parsing_output_json,
            instruction=message,
            images=images
        )
        output_images = self.execution_agent(
            strategy_planning_output_json=strategy_planning_output_json,
            instruction=message,
            images=images
        )
        if return_markdown:
            illustrated_article = self.markdown_agent(
                instruction_parsing_output_json=instruction_parsing_output_json,
                execution_output_images=output_images,
                instruction=message,
                images=images
            )
            return illustrated_article
        else:
            return output_images
