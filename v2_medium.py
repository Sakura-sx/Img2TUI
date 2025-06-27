import os
from PIL import Image
import numpy as np

def v2_medium(img, x, y):
    aspect = img.width / img.height
    term_aspect = x / (y*2)

    if aspect < term_aspect:
        img = img.resize((int(y*aspect*2)*2, y*2))
    if aspect > term_aspect:
        img = img.resize((int(x*2)*2, int(x/aspect)*2))
    
    img_gray = img.convert("L")
    img = img.convert("RGB")

    response = ""

    for i in range(img.height//2):
        for j in range(img.width//2):
            v1 = img_gray.getpixel((j*2, i*2))
            v2 = img_gray.getpixel((j*2+1, i*2))
            v3 = img_gray.getpixel((j*2, i*2+1))
            v4 = img_gray.getpixel((j*2+1, i*2+1))

            vals = sorted([(v1,0),(v2,1),(v3,2),(v4,3)])
            v = [x[0] for x in vals]
            d1 = abs(v[0] - sum(v[1:])/3)
            d2 = abs((v[0]+v[1])/2 - sum(v[2:])/2)
            d3 = abs(v[3] - sum(v[:3])/3)
            winner = max([(d1,[1,1,1,0]),(d2,[1,1,0,0]),(d3,[0,0,0,1])], key=lambda x: x[0])[1]
            out = [0]*4
            for idx, b in zip([x[1] for x in vals], winner): 
                out[idx] = b

            # [1, ?, 
            # ?, ?]
            if out[0] != out[1]:
                # [1, 0, 
                # ?, ?]
                if out[1] == out[2]:
                    # [1, 0, 
                    # 0, ?]
                    if out[2] == out[3]:
                        # [1, 0, 
                        # 0, 0]
                        # ▘ Top left
                        char = "\u2598"
                        fg = img.getpixel((j*2, i*2))
                        bg_colors = np.array([img.getpixel((j*2+1, i*2)), img.getpixel((j*2, i*2+1)), img.getpixel((j*2+1, i*2+1))])
                        bg = tuple((np.mean(bg_colors, axis=0)).astype(int))
                    else:
                        # [1, 0, 
                        # 0, 1]
                        # ▚ Top left and bottom right
                        char = "\u259A"
                        fg_colors = np.array([img.getpixel((j*2, i*2)), img.getpixel((j*2+1, i*2+1))])
                        fg = tuple((np.mean(fg_colors, axis=0)).astype(int))
                        bg_colors = np.array([img.getpixel((j*2, i*2+1)), img.getpixel((j*2+1, i*2))])
                        bg = tuple((np.mean(bg_colors, axis=0)).astype(int))
                else:
                    # [1, 0, 
                    # 1, ?]
                    if out[2] == out[3]:
                        # [1, 0, 
                        # 1, 1]
                        # ▝ Upper right
                        char = "\u259D"
                        fg = img.getpixel((j*2+1, i*2))
                        bg_colors = np.array([img.getpixel((j*2, i*2)), img.getpixel((j*2, i*2+1)), img.getpixel((j*2+1, i*2+1))])
                        bg = tuple((np.mean(bg_colors, axis=0)).astype(int))
                    else:
                        # [1, 0, 
                        # 1, 0]
                        # ▌ Left half
                        char = "\u258C"
                        fg_colors = np.array([img.getpixel((j*2, i*2)), img.getpixel((j*2, i*2+1))])
                        fg = tuple((np.mean(fg_colors, axis=0)).astype(int))
                        bg_colors = np.array([img.getpixel((j*2+1, i*2)), img.getpixel((j*2+1, i*2+1))])
                        bg = tuple((np.mean(bg_colors, axis=0)).astype(int))
            else:
                # [1, 1, 
                # ?, ?]
                if out[1] == out[2]:
                    # [1, 1, 
                    # 1, 0]
                    # ▗ Bottom right
                    char = "\u2597"
                    fg = img.getpixel((j*2+1, i*2+1))
                    bg_colors = np.array([img.getpixel((j*2, i*2)), img.getpixel((j*2, i*2+1)), img.getpixel((j*2+1, i*2))])
                    bg = tuple((np.mean(bg_colors, axis=0)).astype(int))
                else:
                    # [1, 1, 
                    # 0, ?]
                    if out[2] == out[3]:
                        # [1, 1, 
                        # 0, 0]
                        # ▄ Lower half
                        char = "\u2584"
                        fg_colors = np.array([img.getpixel((j*2, i*2+1)), img.getpixel((j*2+1, i*2+1))])
                        fg = tuple((np.mean(fg_colors, axis=0)).astype(int))
                        bg_colors = np.array([img.getpixel((j*2, i*2)), img.getpixel((j*2+1, i*2))])
                        bg = tuple((np.mean(bg_colors, axis=0)).astype(int))
                    else:
                        # [1, 1, 
                        # 0, 1]
                        # ▖ Bottom left
                        char = "\u2596"
                        fg = img.getpixel((j*2, i*2+1))
                        bg_colors = np.array([img.getpixel((j*2, i*2)), img.getpixel((j*2+1, i*2)), img.getpixel((j*2+1, i*2+1))])
                        bg = tuple((np.mean(bg_colors, axis=0)).astype(int))

            response += f"\033[38;2;{fg[0]};{fg[1]};{fg[2]}m\033[48;2;{bg[0]};{bg[1]};{bg[2]}m{char}"

        response += "\033[0m\n"

    return response

if __name__ == "__main__":
    img = Image.open("img.png")
    x, y = os.get_terminal_size()
    print(v2_medium(img, x, y))