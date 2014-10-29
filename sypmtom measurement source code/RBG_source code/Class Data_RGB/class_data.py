import numpy as np
from PIL import Image
import xlwt

# load Data
DATA = np.loadtxt('14.csv',usecols=(0,1), delimiter=',')

x=DATA.shape
print x[0]

# load image
im = Image.open("14.png")
rgb_im = im.convert('1')

# create workbook and worksheet
wbk = xlwt.Workbook()
sheet = wbk.add_sheet('testData')
col = 0

for i in range(0,x[0]):
    pix = rgb_im.getpixel((DATA[i][0] , DATA[i][1]))
    
    
    if pix==255:
        
        sheet.write(i,col,1)
    else:
        sheet.write(i,col,pix)
        


wbk.save('data14.xls')
