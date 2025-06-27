import os
from PIL import Image

def v2_fastest(img, x, y):
    aspect = img.width / img.height
    term_aspect = x / (y*2)

    if aspect < term_aspect:
        img = img.resize((int(y*aspect*2), y))
    if aspect > term_aspect:
        img = img.resize((int(x*2), int(x/aspect)))

    img = img.convert("RGB")

    response = ""

    for i in range(img.height):
        for j in range(img.width):
            response += f"\033[38;2;{img.getpixel((j, i))[0]};{img.getpixel((j, i))[1]};{img.getpixel((j, i))[2]}m█"
        response += "\033[0m\n"

    return response

if __name__ == "__main__":
    img = Image.open("img.png")
    x, y = os.get_terminal_size()
    print(v2_fastest(img, x, y))
