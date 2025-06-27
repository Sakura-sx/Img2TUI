import os
from PIL import Image

def v2_faster(img, x, y):
    aspect = img.width / img.height
    term_aspect = x / (y*2)

    if aspect < term_aspect:
        img = img.resize((int(y*aspect*2), y*2))
    if aspect > term_aspect:
        img = img.resize((int(x*2), int(x/aspect)*2))
    
    img_gray = img.convert("L")
    img = img.convert("RGB")

    response = ""

    for i in range(img.height//2):
        for j in range(img.width):
            fg = img.getpixel((j, i*2))
            bg = img.getpixel((j, i*2+1))
            # ▀ Upper half
            char = "\u2580"
            response += f"\033[38;2;{fg[0]};{fg[1]};{fg[2]}m\033[48;2;{bg[0]};{bg[1]};{bg[2]}m{char}"

        response += "\033[0m\n"

    return response

if __name__ == "__main__":
    img = Image.open("img.png")
    x, y = os.get_terminal_size()
    print(v2_faster(img, x, y))