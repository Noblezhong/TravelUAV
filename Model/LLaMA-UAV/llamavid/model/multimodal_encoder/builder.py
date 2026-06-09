import os
from .clip_encoder import CLIPVisionTower
from .eva_vit import EVAVisionTowerLavis


LLAMAVID_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def _resolve_local_path(path_value):
    if not isinstance(path_value, str) or not path_value:
        return path_value
    if os.path.exists(path_value):
        return path_value
    candidate = os.path.abspath(os.path.join(LLAMAVID_ROOT, path_value))
    if os.path.exists(candidate):
        return candidate
    return path_value


def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    image_processor = getattr(vision_tower_cfg, 'image_processor', getattr(vision_tower_cfg, 'image_processor', "./model_zoo/OpenAI/clip-vit-large-patch14"))
    vision_tower = _resolve_local_path(vision_tower)
    image_processor = _resolve_local_path(image_processor)
    is_absolute_path_exists = os.path.exists(vision_tower)
    
    if not is_absolute_path_exists:
        raise ValueError(f'Not find vision tower: {vision_tower}')
    
    if "openai" in vision_tower.lower() or "laion" in vision_tower.lower():
        return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    elif "lavis" in vision_tower.lower() or "eva" in vision_tower.lower():
        return EVAVisionTowerLavis(vision_tower, image_processor, args=vision_tower_cfg, **kwargs)
    else:
        raise ValueError(f'Unknown vision tower: {vision_tower}')
    
