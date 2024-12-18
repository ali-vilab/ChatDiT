import re
import torch
import numpy as np
from diffusers import FluxPipeline, FluxInpaintPipeline, FluxFillPipeline
from PIL import Image

__all__ = ['InContextPipeline']


class InContextPipeline:

    def __init__(
        self,
        model_name_or_path='black-forest-labs/FLUX.1-dev',
        lora_name_or_path=None,
        fill_model_name_or_path=None,
        dtype=torch.bfloat16,
        device=torch.device('cuda:0')
    ):
        self.model_name_or_path = model_name_or_path
        self.lora_name_or_path = lora_name_or_path
        self.fill_model_name_or_path = fill_model_name_or_path
        self.dtype = dtype
        self.device = device

        # generation pipeline
        self.pipe = FluxPipeline.from_pretrained(model_name_or_path, torch_dtype=dtype).to(device)
        if lora_name_or_path:
            self.pipe.load_lora_weights(lora_name_or_path)

        # fill pipeline
        if fill_model_name_or_path is None:
            self.inpaint_pipe = FluxInpaintPipeline(
                scheduler=self.pipe.scheduler,
                vae=self.pipe.vae,
                text_encoder=self.pipe.text_encoder,
                tokenizer=self.pipe.tokenizer,
                text_encoder_2=self.pipe.text_encoder_2,
                tokenizer_2=self.pipe.tokenizer_2,
                transformer=self.pipe.transformer
            )
        else:
            self.inpaint_pipe = FluxFillPipeline.from_pretrained(fill_model_name_or_path, torch_dtype=dtype).to(device)

    def __call__(
        self,
        prompt,
        prompt_2=None,
        images=[],
        num_outputs=1,
        height=None,
        width=None,
        num_inference_steps=28,
        guidance_scale=None,
        generator=None,
        latents=None,
        max_sequence_length=512,
        preprocess_type='resize_and_pad',
        reformat_prompt=False,
        border_size=0,
        border_color='black'
    ):
        # check prompt
        if prompt_2 is None:
            prompt_2 = prompt
        
        # check input and output counts
        num_inputs = len(images)
        assert num_outputs >= 1
        num_panels = num_inputs + num_outputs
        
        # check size
        if height is None:
            height = 1024
        if width is None:
            width = 1024

        # check guidance scale
        if guidance_scale is None:
            if len(images) == 0:
                guidance_scale = 3.5
            elif isinstance(self.inpaint_pipe, FluxFillPipeline):
                guidance_scale = 30
            else:
                guidance_scale = 7.0
        
        # check preprocess_type
        assert preprocess_type in ('resize_and_pad', 'resize_and_crop')
        preprocess_fn = {
            'resize_and_pad': self._resize_and_pad,
            'resize_and_crop': self._resize_and_crop
        }[preprocess_type]
        
        # optimize panel layout
        if num_panels == 1:
            rows, cols = 1, 1
        else:
            rows, cols, prompt, prompt_2 = self._optimize_panel_layout(
                num_panels, height, width, prompt, prompt_2, reformat_prompt
            )
        
        # inference
        if num_inputs == 0:
            if latents is not None:
                latents = latents.to(self.pipe.device)
            grid = self.pipe(
                prompt=prompt,
                prompt_2=prompt_2,
                height=height * rows,
                width=width * cols,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                num_images_per_prompt=1,
                generator=generator,
                latents=latents,
                output_type='pil',
                max_sequence_length=max_sequence_length
            ).images[0]
        else:
            if latents is not None:
                latents = latents.to(self.inpaint_pipe.device)
            
            # create the concatenated big image
            panels = [preprocess_fn(
                u.convert('RGB'), height, width, border_size, border_color
            ) for u in images]
            panels += [Image.new('RGB', (width, height), (127, 127, 127))] * num_outputs
            grid = self._make_grid(panels, rows, cols)

            # create the big mask
            mask = Image.new('L', grid.size, 255)
            for i in range(num_inputs):
                row, col = i // cols, i % cols
                mask.paste(Image.new('L', (width, height), 0), (width * col, height * row))
            
            # inference
            kwargs = {'strength': 1.0} if isinstance(self.inpaint_pipe, FluxInpaintPipeline) else {}
            grid = self.inpaint_pipe(
                prompt=prompt,
                prompt_2=prompt_2,
                image=grid,
                mask_image=mask,
                height=height * rows,
                width=width * cols,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                num_images_per_prompt=1,
                generator=generator,
                output_type='pil',
                max_sequence_length=max_sequence_length,
                **kwargs
            ).images[0]
        
        # postprocess
        if num_panels == 1:
            panels = [grid]
        else:
            panels = self._split_grid(grid, rows, cols)
        return panels[-num_outputs:]

    def _optimize_panel_layout(self, num_panels, height, width, prompt, prompt_2, reformat_prompt):
        # check num_panels
        assert num_panels >= 1 and num_panels <= 12, 'Current we only support num_panels between 1 and 12'
        if num_panels == 1:
            return 1, 1, prompt, prompt_2
        
        # optimize panel layout to achieve an aspect ratio closest to 1.0
        best_aspect_ratio = float('inf')
        best_layout = None
        for rows in range(1, num_panels + 1):
            if num_panels % rows == 0:
                cols = num_panels // rows
                grid_height, grid_width = height * rows, width * cols
                aspect_ratio = max(grid_height / grid_width, grid_width / grid_height)
                if aspect_ratio < best_aspect_ratio:
                    best_aspect_ratio = aspect_ratio
                    best_layout = (rows, cols)
        rows, cols = best_layout

        # reformat prompts
        if reformat_prompt:
            assert num_panels <= 12, 'Only supports reformat_prompt=True with <= 12 panels'

            # semantic names
            number_to_name = {
                2: 'TWO',
                3: 'THREE',
                4: 'FOUR',
                5: 'FIVE',
                6: 'SIX',
                7: 'SEVEN',
                8: 'EIGHT',
                9: 'NINE',
                10: 'TEN',
                11: 'ELEVEN',
                12: 'TWELVE'
            }
            rows_to_names = {
                2: ['TOP', 'BOTTOM'],
                3: ['TOP', 'MIDDLE', 'BOTTOM'],
            }
            cols_to_names = {
                2: ['LEFT', 'RIGHT'],
                3: ['LEFT', 'MIDDLE', 'RIGHT']
            }

            # target panel names
            target_names = [number_to_name[num_panels] + '-PANEL']
            if rows <= 3 and cols <= 3:
                if rows == 1:
                    target_names += cols_to_names[cols]
                elif cols == 1:
                    target_names += rows_to_names[rows]
                else:
                    for row_name in rows_to_names[rows]:
                        for col_name in cols_to_names[cols]:
                            target_names.append(f'{row_name}-{col_name}')
            else:
                target_names += [f'PANEL-{i + 1}' for i in range(num_panels)]
            
            # name patterns
            pattern = r'\[((?:TWO|THREE|FOUR|FIVE|SIX|SEVEN|EIGHT|NINE|TEN|ELEVEN|TWELVE|MULTI|PANEL|[0-9]|-)+)\]'
            
            # process prompt
            source_names = re.findall(pattern, prompt)
            assert len(source_names) == len(target_names) == 1 + num_panels
            for src_name, tar_name in zip(source_names, target_names):
                prompt = prompt.replace(src_name, tar_name)
            
            # process prompt_2
            source_names = re.findall(pattern, prompt_2)
            assert len(source_names) == len(target_names) == 1 + num_panels
            for src_name, tar_name in zip(source_names, target_names):
                prompt_2 = prompt_2.replace(src_name, tar_name)
        return rows, cols, prompt, prompt_2

    def _make_grid(self, panels, rows, cols):
        assert [u.size == panels[0].size for u in panels]
        assert len(panels) == rows * cols and rows >= 1 and cols >= 1

        # init blank grid
        width, height = panels[0].size
        grid = Image.new(panels[0].mode, (width * cols, height * rows))

        # paste panels
        for i, panel in enumerate(panels):
            row, col = i // cols, i % cols
            grid.paste(panel, (width * col, height * row))
        return grid
    
    def _split_grid(self, grid, rows, cols):
        height = grid.height // rows
        width = grid.width // cols
        panels = []
        for i in range(rows):
            for j in range(cols):
                panels.append(grid.crop((
                    j * width,
                    i * height,
                    (j + 1) * width,
                    (i + 1) * height
                )))
        return panels

    def _resize_and_pad(self, image, height, width, border_size=0, border_color='black'):
        # resize
        scale = min(height / image.height, width / image.width)
        image = image.resize((int(image.width * scale), int(image.height * scale)), Image.LANCZOS)

        # pad (with average color)
        color = tuple(np.mean(np.array(image), axis=(0, 1)).astype(int))
        pad_image = Image.new(image.mode, (width, height), color)
        pad_image.paste(image, ((width - image.width) // 2, (height - image.height) // 2))
        image = pad_image

        # add borders
        if border_size > 0:
            new_image = Image.new(image.mode, image.size, border_color)
            new_image.paste(
                image.crop((border_size, border_size, image.width - border_size, image.height - border_size)),
                (border_size, border_size)
            )
            image = new_image
        return image
    
    def _resize_and_crop(self, image, height, width, border_size=0, border_color='black'):
        # resize
        scale = max(height / image.height, width / image.width)
        image = image.resize((int(image.width * scale), int(image.height * scale)), Image.LANCZOS)

        # center crop
        image = image.crop((
            (image.width - width) // 2,
            (image.height - height) // 2,
            (image.width - width) // 2 + width,
            (image.height - height) // 2 + height
        ))

        # add borders
        if border_size > 0:
            new_image = Image.new(image.mode, image.size, border_color)
            new_image.paste(
                image.crop((border_size, border_size, image.width - border_size, image.height - border_size)),
                (border_size, border_size)
            )
            image = new_image
        return image
