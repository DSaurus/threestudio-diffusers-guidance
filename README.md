# threestudio-shap-e
This is diffusers guidance extension of threestudio.

## Installation
```
cd custom
git clone https://github.com/DSaurus/threestudio-diffusers-guidance
```

## Examples
```
python launch.py --config custom/threestudio-diffusers-guidance/configs/latentnerf-sd-xl.yaml --train --gpu 0 system.prompt_processor.prompt="a delicious hamburger"

python launch.py --config custom/threestudio-diffusers-guidance/configs/dreamfusion-sd.yaml --train --gpu 0 system.prompt_processor.prompt="a delicious hamburger"

python launch.py --config custom/threestudio-diffusers-guidance/configs/dreamfusion-if.yaml --train --gpu 0 system.prompt_processor.prompt="a delicious hamburger"

python launch.py --config custom/threestudio-diffusers-guidance/configs/dreamfusion-sd-ism.yaml --train --gpu 0 system.prompt_processor.prompt="a delicious hamburger"
```
