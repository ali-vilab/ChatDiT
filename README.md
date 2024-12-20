# ChatDiT

🔥 **Latest News!**

- **[2024-12-18]** 📂 Code for **[ChatDiT](https://arxiv.org/abs/2412.12571)** are now available!
- **[2024-12-18]** 📜 Our latest paper, **[ChatDiT](https://arxiv.org/abs/2412.12571)**, introduces a training-free, zero-shot, general-purpose, and interactive image generation system built upon diffusion transformers.

Welcome to the official repository for **ChatDiT: A Training-Free Baseline for Task-Agnostic Free-Form Chatting with Diffusion Transformers** ([Paper](https://arxiv.org/abs/2412.12571) | [Project Page](https://ali-vilab.github.io/ChatDiT-Page/)).

ChatDiT supports seamless multi-round, free-form conversations with DiTs. It supports referencing zero to multiple images to generate a new set of images, or, if desired, a fully illustrated article in response.

Dispite its simplicity and training-free design, ChatDiT achieves the best performance (**23.19**) on [IDEA-Bench](https://ali-vilab.github.io/IDEA-Bench-Page/), surpassing models like EMU2 (**6.8**) and OmniGen (**6.47**).

> Note: ChatDiT is still a prototype, so some features may not work as expected. However, it provides a robust baseline for future research and development.

## Getting Started

### Prerequisites

Install [PyTorch](https://pytorch.org/) based on the [official guidelines](https://pytorch.org/get-started/locally/), then install diffusers and openai:

```bash
pip install diffusers openai
```

### Simple Usage

ChatDiT offers a simple, intuitive interface for interacting with both text and images.

```python
import openai
import torch
from chatdit import ChatDiT

app = ChatDiT(
    client=openai.OpenAI(),
    device=torch.device('cuda')
)

# text-to-image
image = app.chat('Generate a dog wearing sunglasses swimming in the pool')[0]

# text-to-image
images = app.chat('Show two different views of a 3D black cat blind box')

# image-to-image
image = app.chat('Create a cup with the abstract character wrapping around it', images=[input_image])[0]

# image-to-images
images = app.chat('Place the product in 4 different scenarios showcasing its usages', images=[input_image])

# images-to-image
image = app.chat(
    'Create a 3D render of the product based on its three-view images',
    images=[front_image, back_image, side_image]
)

# images-to-images
images = app.chat(
    'Show two interaction scenarios of the two characters',
    images=[character1_image, character2_image]
)

# interleaved text-image article
article = app.chat(
    'Tell the story of an architect designing and constructing an innovative building. Images should show the various stages from initial sketches to completed building.',
    return_markdown=True
)
article.save('./innovative_building_designing/')  # saves both markdown and images
```

Checkout input/output cases in our [Project Page](https://ali-vilab.github.io/ChatDiT-Page/).

## License

This repository is licensed under the [MIT LICENSE](./LICENSE). It uses [FLUX](https://github.com/black-forest-labs/flux) as the base model, so users must also comply with FLUX's license. For more details, refer to [FLUX's License](https://github.com/black-forest-labs/flux/tree/main/model_licenses).

## Citation

If you use ChatDiT in your research, please cite our paper:

```bibtex
@article{lhhuang2024chatdit,
  title={ChatDiT: A Training-Free Baseline for Task-Agnostic Free-Form Chatting with Diffusion Transformers},
  author={Huang, Lianghua and Wang, Wei and Wu, Zhi-Fan and Shi, Yupeng and Liang, Chen and Shen, Tong and Zhang, Han and Dou, Huanzhang and Liu, Yu and Zhou, Jingren},
  booktitle={arXiv preprint arxiv:2412.12571},
  year={2024}
}
```

Also, please cite the following papers for related work:

```bibtex
@article{lhhuang2024iclora,
  title={In-Context LoRA for Diffusion Transformers},
  author={Huang, Lianghua and Wang, Wei and Wu, Zhi-Fan and Shi, Yupeng and Dou, Huanzhang and Liang, Chen and Feng, Yutong and Liu, Yu and Zhou, Jingren},
  journal={arXiv preprint arxiv:2410.23775},
  year={2024}
}
```

```bibtex
@article{lhhuang2024groupdiffusion,
  title={Group Diffusion Transformers are Unsupervised Multitask Learners},
  author={Huang, Lianghua and Wang, Wei and Wu, Zhi-Fan and Dou, Huanzhang and Shi, Yupeng and Feng, Yutong and Liang, Chen and Liu, Yu and Zhou, Jingren},
  journal={arXiv preprint arxiv:2410.15027},
  year={2024}
}
```