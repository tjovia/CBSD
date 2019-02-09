import Image
im = Image.open('2.png')
rgb_im = im.convert('RGB')
r, g, b = rgb_im.getpixel((0, 0))

print r, g, b
