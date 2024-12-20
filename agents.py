import openai
import io
import base64
import os
import json
import re
import numpy as np
import torch
from PIL import Image

from in_context_pipeline import InContextPipeline

__all__ = [
    'InstructionParsingAgent',
    'StrategyPlanningAgent',
    'ExecutionAgent',
    'MarkdownAgent'
]

SYSTEM_FOLDER = './system'


class Agent:
    """
    Metaclass of agents.
    """
    def __call__(self, *args, retry=5, verbose=True, **kwargs):
        name = self.__class__.__name__
        exception = None
        for i in range(retry):
            if verbose and i > 0:
                print(f'[{name}] Retrying [{i + 1}/{retry}]...')
            try:
                if verbose:
                    print(f'Running {name}')
                return self.action(*args, **kwargs)
            except Exception as e:
                exception = e
        else:
            raise exception
    
    def action(self, *args, **kwargs):
        raise NotImplementedError

    def send_request(
        self,
        client,
        message,
        images=[],
        history=[],
        model='gpt-4o-2024-08-06',
        max_tokens=8192,
        response_format={'type': 'json_object'},
        **kwargs
    ):
        assert isinstance(client, openai.OpenAI)

        # prepare messages
        if not images:
            content = message
        else:
            images = [self.encode_image(u) for u in images]
            content = [{
                'type': 'image_url',
                'image_url': {'url': f'data:image/jpeg;base64,{u}', 'detail': 'high'}
            } for u in images]
            content += [{'type': 'text', 'text': message}]
        messages = history + [{'role': 'user', 'content': content}]
        
        # send request
        message = client.chat.completions.create(
            messages=messages,
            model=model,
            response_format=response_format,
            max_tokens=max_tokens,
            **kwargs
        ).choices[0].message
        return message.content
    
    def encode_image(self, image, max_side=2048):
        if max(image.size) > max_side:
            scale = max_side / max(image.size)
            image = image.resize((int(image.width * scale), int(image.height * scale)), Image.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG')
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode('utf-8')


#---------------------- InstructionParsingAgent ----------------------#

class DescriptionAgent(Agent):
    """
    Create descriptions for user's input images.
    """
    def __init__(self, client, system_path=os.path.join(SYSTEM_FOLDER, 'description_agent.md')):
        self.client = client
        with open(system_path) as f:
            self.system = f.read()
    
    def action(self, instruction, images=[]):
        # if no uploaded images, return empty descriptions
        if not images:
            return {
                'explanation': None,
                'descriptions': {'overview': None, 'input_images': []}
            }
        
        # prepare inputs
        input_json = {
            'instruction': instruction,
            'input_image_count': len(images)
        }

        # send request
        output_json = json.loads(self.send_request(
            client=self.client,
            message=json.dumps(input_json, indent=4, ensure_ascii=False),
            images=images,
            history=[{'role': 'system', 'content': self.system}],
            response_format={'type': 'json_object'}
        ))

        # check outputs
        assert output_json.keys() == {'explanation', 'descriptions'}
        assert output_json['descriptions'].keys() == {'overview', 'input_images'}
        input_images = output_json['descriptions']['input_images']
        assert len(input_images) == len(images)
        assert all(u.keys() == {'image_id', 'description'} for u in input_images)
        assert all(f'_{i + 1:04d}_' in u['image_id'] for i, u in enumerate(input_images))
        return output_json


class CountingAgent(Agent):
    """
    Predict the number of desired output images from user's instruction.
    """
    def __init__(self, client, system_path=os.path.join(SYSTEM_FOLDER, 'counting_agent.md')):
        self.client = client
        with open(system_path) as f:
            self.system = f.read()

    def action(self, instruction, images=[]):
        # prepare inputs
        input_json = {
            'instruction': instruction,
            'input_image_count': len(images)
        }

        # send request
        output_json = json.loads(self.send_request(
            client=self.client,
            message=json.dumps(input_json, indent=4, ensure_ascii=False),
            images=[],
            history=[{'role': 'system', 'content': self.system}],
            response_format={'type': 'json_object'}
        ))

        # check outputs
        assert output_json.keys() == {'explanation', 'input_image_count', 'output_image_count'}
        assert output_json['input_image_count'] == input_json['input_image_count']
        assert output_json['output_image_count'] >= 1
        return output_json


class PromptingAgent(Agent):
    """
    Create descriptions for desired output images.
    """
    def __init__(
        self,
        client,
        system_text_to_image_path=os.path.join(SYSTEM_FOLDER, 'prompting_text_to_image_agent.md'),
        system_text_to_images_path=os.path.join(SYSTEM_FOLDER, 'prompting_text_to_images_agent.md'),
        system_images_to_images_path=os.path.join(SYSTEM_FOLDER, 'prompting_images_to_images_agent.md')
    ):
        self.client = client
        with open(system_text_to_image_path) as f:
            self.system_text_to_image = f.read()
        with open(system_text_to_images_path) as f:
            self.system_text_to_images = f.read()
        with open(system_images_to_images_path) as f:
            self.system_images_to_images = f.read()
    
    def action(
        self,
        counting_output_json,
        description_output_json,
        instruction,
        images=[]
    ):
        input_image_count = counting_output_json['input_image_count']
        output_image_count = counting_output_json['output_image_count']
        assert input_image_count == len(images) == len(description_output_json['descriptions']['input_images'])

        # prepare inputs
        if input_image_count == 0:
            input_json = {
                'instruction': instruction,
                'input_image_count': input_image_count,
                'output_image_count': output_image_count
            }
            system = self.system_text_to_image if output_image_count == 1 else self.system_text_to_images
        else:
            input_json = {
                'instruction': instruction,
                'input_image_count': input_image_count,
                'output_image_count': output_image_count,
                'descriptions': {
                    'input_images_overview': description_output_json['descriptions']['overview'],
                    'input_images': description_output_json['descriptions']['input_images']
                }
            }
            system = self.system_images_to_images
        
        # send requests
        output_json = json.loads(self.send_request(
            client=self.client,
            message=json.dumps(input_json, indent=4, ensure_ascii=False),
            images=images,
            history=[{'role': 'system', 'content': system}],
            response_format={'type': 'json_object'}
        ))

        # post-process
        if input_image_count == 0 and output_image_count == 1:
            output_json = {
                'explanation': output_json['explanation'],
                'descriptions': {
                    'overview': None,
                    'output_images': [{
                        'image_id': output_json['image_id'],
                        'description': output_json['prompt']
                    }]
                }
            }
        
        # check outputs
        assert output_json.keys() == {'explanation', 'descriptions'}
        assert output_json['descriptions'].keys() == {'overview', 'output_images'}
        output_images = output_json['descriptions']['output_images']
        assert len(output_images) == output_image_count
        assert all(u.keys() == {'image_id', 'description'} for u in output_images)
        assert all(f'_{i + 1:04d}_' in u['image_id'] for i, u in enumerate(output_images))
        return output_json


class InstructionParsingAgent(Agent):

    def __init__(self, client):
        self.description_agent = DescriptionAgent(client=client)
        self.counting_agent = CountingAgent(client=client)
        self.prompting_agent = PromptingAgent(client=client)
    
    def __call__(self, *args, **kwargs):
        if 'retry' not in kwargs:
            kwargs['retry'] = 1
        return super().__call__(*args, **kwargs)
    
    def action(self, instruction, images=[]):
        # instruction parsing
        description_output_json = self.description_agent(instruction, images)
        counting_output_json = self.counting_agent(instruction, images)
        prompting_output_json = self.prompting_agent(
            counting_output_json=counting_output_json,
            description_output_json=description_output_json,
            instruction=instruction,
            images=images
        )

        # organize outputs
        output_json = {
            'input_image_count': counting_output_json['input_image_count'],
            'output_image_count': counting_output_json['output_image_count'],
            'descriptions': {
                'input_images': description_output_json['descriptions']['input_images'],
                'output_images': prompting_output_json['descriptions']['output_images']
            }
        }
        return output_json


#---------------------- StrategyPlanningAgent ----------------------#

class ReferencingAgent(Agent):
    """
    Reference inputs images for each desired output image.
    """
    def action(
        self,
        instruction_parsing_output_json,
        instruction,
        images=[]
    ):
        # parse counts
        input_image_count = instruction_parsing_output_json['input_image_count']
        output_image_count = instruction_parsing_output_json['output_image_count']

        # parse descriptions
        descriptions = instruction_parsing_output_json['descriptions']
        assert input_image_count == len(images) == len(descriptions['input_images'])
        input_ids = [u['image_id'] for u in descriptions['input_images']]
        output_ids = [u['image_id'] for u in descriptions['output_images']]

        # strategy routing
        if input_image_count == 0 and output_image_count <= 4:
            groups = [{
                'input_image_ids': [],
                'output_image_ids': output_ids
            }]
        elif input_image_count == 0 and output_image_count > 4:
            groups = [{
                'input_image_ids': [],
                'output_image_ids': output_ids[:4]
            }] + [{
                'input_image_ids': output_ids[:3],
                'output_image_ids': [u]
            } for u in output_ids[4:]]
        elif input_image_count == 1 and output_image_count == 1:
            groups = [{
                'input_image_ids': input_ids,
                'output_image_ids': output_ids
            }]
        elif input_image_count == 1 and output_image_count > 1:
            groups = [{
                'input_image_ids': input_ids + output_ids[:i],
                'output_image_ids': [output_ids[i]]
            } for i in range(output_image_count)]
        elif input_image_count > 1 and output_image_count == 1:
            groups = [{
                'input_image_ids': input_ids,
                'output_image_ids': output_ids
            }]
        elif input_image_count > 1 and output_image_count > 1:
            groups =[{
                'input_image_ids': input_ids,
                'output_image_ids': [u]
            } for u in output_ids]
        else:
            raise NotImplementedError('Impossible!')
        
        # check outputs
        assert all(u.keys() == {'input_image_ids', 'output_image_ids'} for u in groups)
        assert all(set(u['input_image_ids']) & set(u['output_image_ids']) == set() for u in groups)
        assert [t for u in groups for t in u['output_image_ids']] == output_ids
        merged_ids = input_ids + output_ids
        if input_ids:
            assert all(
                max(merged_ids.index(t) for t in u['input_image_ids']) <
                min(merged_ids.index(t) for t in u['output_image_ids'])
                for u in groups
            )
        return groups


class PanelizingAgent(Agent):
    """
    Create in-context multi-panel prompt for each group of input and output images.
    """
    def __init__(self, client, system_path=os.path.join(SYSTEM_FOLDER, 'panelizing_agent.md')):
        self.client = client
        with open(system_path) as f:
            self.system = f.read()
    
    def __call__(self, *args, **kwargs):
        if 'retry' not in kwargs:
            kwargs['retry'] = 1
        return super().__call__(*args, **kwargs)
    
    def action(
        self,
        instruction_parsing_output_json,
        referencing_output_json,
        instruction,
        images=[]
    ):
        # parse instruction-parsing results
        input_image_count = instruction_parsing_output_json['input_image_count']
        output_image_count = instruction_parsing_output_json['output_image_count']
        descriptions = instruction_parsing_output_json['descriptions']
        descriptions = descriptions['input_images'] + descriptions['output_images']

        # parse referencing results
        id2desc = {u['image_id']: u for u in descriptions}
        groups = [[
            id2desc[t] for t in u['input_image_ids'] + u['output_image_ids']
        ] for u in referencing_output_json]

        # prepare inputs
        input_jsons = [{
            'instruction': instruction,
            'input_image_count': input_image_count,
            'output_image_count': output_image_count,
            'descriptions': descriptions,
            'panels': {f'panel_{i + 1}': u for i, u in enumerate(group)}
        } for group in groups]

        # send requests
        output_jsons = [self._create_prompt(u) for u in input_jsons]
        return output_jsons
    
    def _create_prompt(self, input_json, retry=5, verbose=True):
        exception = None
        for i in range(retry):
            if verbose and i > 0:
                print(f'[Inner {self.__class__.__name__}] Retrying [{i + 1}/{retry}]...')
            try:
                # simple text-to-image
                if len(input_json['panels']) == 1:
                    return {'explanation': None, 'prompt': input_json['panels'][0]['description']}

                # send request
                output_json = json.loads(self.send_request(
                    client=self.client,
                    message=json.dumps(input_json, indent=4, ensure_ascii=False),
                    images=[],
                    history=[{'role': 'system', 'content': self.system}],
                    response_format={'type': 'json_object'}
                ))

                # check outputs
                pattern = r'\[((?:TWO|THREE|FOUR|FIVE|SIX|SEVEN|EIGHT|NINE|TEN|ELEVEN|TWELVE|MULTI|PANEL|[0-9]|-)+)\]'
                names = re.findall(pattern, output_json['prompt'])
                assert len(names) == len(input_json['panels']) + 1
                assert names[1:] == [f'PANEL-{i + 1}' for i in range(len(input_json['panels']))]
                return output_json
            except Exception as e:
                exception = e
        else:
            raise exception


class StrategyPlanningAgent(Agent):

    def __init__(self, client):
        self.referencing_agent = ReferencingAgent()
        self.panelizing_agent = PanelizingAgent(client=client)
    
    def __call__(self, *args, **kwargs):
        if 'retry' not in kwargs:
            kwargs['retry'] = 1
        return super().__call__(*args, **kwargs)
    
    def action(
        self,
        instruction_parsing_output_json,
        instruction,
        images=[]
    ):
        # run agents
        referencing_output_json = self.referencing_agent(
            instruction_parsing_output_json,
            instruction,
            images
        )
        panelizing_output_json = self.panelizing_agent(
            instruction_parsing_output_json,
            referencing_output_json,
            instruction,
            images
        )

        # organize outputs
        image_ids = [u['image_id'] for u in (
            instruction_parsing_output_json['descriptions']['input_images'] +
            instruction_parsing_output_json['descriptions']['output_images']
        )]
        output_json = {
            'steps': [{
                'prompt': v['prompt'],
                'images': [image_ids.index(t) for t in u['input_image_ids']],
                'num_outputs': len(u['output_image_ids'])
            } for u, v in zip(referencing_output_json, panelizing_output_json)]
        }
        return output_json


#---------------------- ExecutionAgent ----------------------#

class ExecutionAgent:
    
    def __init__(self, **kwargs):
        self.pipe = InContextPipeline(**kwargs)
    
    def __call__(
        self,
        strategy_planning_output_json,
        instruction,
        images=[],
        seed=2024,
        **kwargs
    ):
        images = images[:]  # avoid inplace modification

        # params
        steps = strategy_planning_output_json['steps']
        input_image_count = len(images)
        aspect_ratio = 1. if input_image_count == 0 else np.power(
            2, np.median(np.log2([u.width / u.height for u in images]))
        )

        # execute steps
        steps = strategy_planning_output_json['steps']
        for step in steps:
            # calculate size
            panel_count = len(step['images']) + step['num_outputs']
            area = 2 ** 20 if panel_count == 1 else (2 ** 21 / panel_count)
            height = int((area / aspect_ratio) ** 0.5) // 64 * 64
            width = int((area * aspect_ratio) ** 0.5) // 64 * 64

            # inference
            images += self.pipe(
                prompt=step['prompt'],
                images=[images[i] for i in step['images']],
                num_outputs=step['num_outputs'],
                height=height,
                width=width,
                generator=torch.Generator(device=self.pipe.device).manual_seed(seed),
                preprocess_type='resize_and_crop',
                reformat_prompt=True,
                **kwargs
            )
        return images[input_image_count:]


#---------------------- MarkdownAgent ----------------------#

class IllustratedArticle:
    
    def __init__(self, markdown, image_dict):
        # check inputs
        pattern = r"\(input_[\w]+\.jpg\)|\(output_[\w]+\.jpg\)"
        image_keys = set(u[1:-5] for u in re.findall(pattern, markdown))
        assert image_keys.issubset(image_dict.keys())

        # assign variables
        self.markdown = markdown
        self.image_dict = image_dict
        self.image_keys = image_keys
    
    def save(self, folder_path, name='illustrated_article'):
        os.makedirs(folder_path, exist_ok=True)

        # save markdown file
        with open(os.path.join(folder_path, name + '.md'), 'w') as f:
            f.write(self.markdown)
        
        # save images
        for k in self.image_keys:
            self.image_dict[k].save(os.path.join(folder_path, k + '.jpg'))


class MarkdownAgent(Agent):
    """
    Create interleaved text-image article from previous agents' outputs.
    """
    def __init__(self, client, system_path=os.path.join(SYSTEM_FOLDER, 'markdown_agent.md')):
        self.client = client
        with open(system_path) as f:
            self.system = f.read()

    def action(
        self,
        instruction_parsing_output_json,
        execution_output_images,
        instruction,
        images=[]
    ):
        # prepare inputs
        input_json = {
            'instruction': instruction,
            'input_image_count': instruction_parsing_output_json['input_image_count'],
            'output_image_count': instruction_parsing_output_json['output_image_count'],
            'descriptions': instruction_parsing_output_json['descriptions']
        }

        # send request
        markdown = self.send_request(
            client=self.client,
            message=json.dumps(input_json, indent=4, ensure_ascii=False),
            images=[],
            history=[{'role': 'system', 'content': self.system}],
            response_format={'type': 'text'}
        )

        # create illustrated article
        image_ids = set(
            [u['image_id'] for u in instruction_parsing_output_json['descriptions']['input_images']] +
            [u['image_id'] for u in instruction_parsing_output_json['descriptions']['output_images']]
        )
        images = images + execution_output_images
        image_dict = {u: v for u, v in zip(image_ids, images)}
        article = IllustratedArticle(markdown, image_dict)
        return article
