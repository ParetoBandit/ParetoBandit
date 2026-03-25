import math
import os
from PIL import Image, ImageDraw, ImageFont

def draw_text_on_curve(image_path, output_path, text, center, radius, font_path, font_size, text_color, position="bottom"):
    img = Image.open(image_path).convert("RGBA")
    
    try:
        font = ImageFont.truetype(font_path, font_size)
    except Exception as e:
        print(f"Font error: {e}")
        font = ImageFont.load_default()
        
    text_layer = Image.new("RGBA", img.size, (0,0,0,0))
    
    char_widths = []
    temp_draw = ImageDraw.Draw(Image.new("RGBA", (1,1)))
    for char in text:
        bbox = temp_draw.textbbox((0, 0), char, font=font)
        char_widths.append(bbox[2] - bbox[0])
        
    tracking = font_size * 0.15
    total_arc_length = sum(char_widths) + tracking * (len(text) - 1)
    
    total_angle = total_arc_length / radius
    
    if position == "top":
        start_angle = -math.pi / 2 - total_angle / 2
    else:
        start_angle = math.pi / 2 + total_angle / 2
        
    current_angle = start_angle
    
    for i, char in enumerate(text):
        w = char_widths[i]
        
        if position == "top":
            char_angle = current_angle + (w / radius) / 2
        else:
            char_angle = current_angle - (w / radius) / 2
            
        char_x = center[0] + radius * math.cos(char_angle)
        char_y = center[1] + radius * math.sin(char_angle)
        
        c_bbox = temp_draw.textbbox((0, 0), char, font=font)
        c_w, c_h = c_bbox[2] - c_bbox[0], c_bbox[3] - c_bbox[1]
        
        char_canvas_size = int(font_size * 3)
        char_img = Image.new("RGBA", (char_canvas_size, char_canvas_size), (0,0,0,0))
        c_draw = ImageDraw.Draw(char_img)
        
        draw_x = (char_canvas_size - c_w) / 2 - c_bbox[0]
        draw_y = (char_canvas_size - c_h) / 2 - c_bbox[1]
        
        # Add a subtle shadow/glow to make text pop
        glow_radius = 2
        for gx in [-glow_radius, 0, glow_radius]:
            for gy in [-glow_radius, 0, glow_radius]:
                c_draw.text((draw_x+gx, draw_y+gy), char, font=font, fill=(0,0,0,150))
        
        c_draw.text((draw_x, draw_y), char, font=font, fill=text_color)
        
        char_angle_deg = math.degrees(char_angle)
        if position == "top":
            rot_deg = - (char_angle_deg + 90)
        else:
            rot_deg = - (char_angle_deg - 90)
            
        char_img_rotated = char_img.rotate(rot_deg, expand=True, resample=Image.BICUBIC)
        
        paste_x = int(char_x - char_img_rotated.width / 2)
        paste_y = int(char_y - char_img_rotated.height / 2)
        
        text_layer.paste(char_img_rotated, (paste_x, paste_y), char_img_rotated)
        
        if position == "top":
            current_angle += (w + tracking) / radius
        else:
            current_angle -= (w + tracking) / radius
            
    final_img = Image.alpha_composite(img, text_layer)
    if img.mode != "RGBA":
        final_img = final_img.convert(img.mode)
    final_img.save(output_path)

base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "blog")
font = "/System/Library/Fonts/Avenir.ttc"

radii = [250, 300, 350, 400]
for r in radii:
    draw_text_on_curve(f"{base_path}/paretobandit_logo.png", 
                       f"{base_path}/paretobandit_logo_text_bottom_r{r}.png", 
                       "ParetoBandit", (512, 512), r, font, 70, (255, 255, 255, 255), "bottom")
    draw_text_on_curve(f"{base_path}/paretobandit_logo.png", 
                       f"{base_path}/paretobandit_logo_text_top_r{r}.png", 
                       "ParetoBandit", (512, 512), r, font, 70, (255, 255, 255, 255), "top")
    
    draw_text_on_curve(f"{base_path}/paretobandit_logo_transparent.png", 
                       f"{base_path}/paretobandit_logo_transparent_text_bottom_r{r}.png", 
                       "ParetoBandit", (512, 512), r, font, 70, (255, 255, 255, 255), "bottom")
    draw_text_on_curve(f"{base_path}/paretobandit_logo_transparent.png", 
                       f"{base_path}/paretobandit_logo_transparent_text_top_r{r}.png", 
                       "ParetoBandit", (512, 512), r, font, 70, (255, 255, 255, 255), "top")

print("Generated variations.")
