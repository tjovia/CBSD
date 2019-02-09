import numpy as np
from PIL import Image
import xlwt

#Load Data
DATA = np.loadtxt('2.csv', 
               usecols=(0,1), delimiter=',')

#print DATA.shape

#print DATA[0:,0]

#print DATA[0:,1]
#print DATA[0][1]
x=DATA.shape
print x[0]

#Load image
im = Image.open("2.png")
rgb_im = im.convert('L')
#print rgb_im.getpixel((0 , 70))

# Create workbook and worksheet
wbk = xlwt.Workbook()
sheet = wbk.add_sheet('testData')
col = 0
#pix = zeros([16065,1])
for i in range(0,x[0]):
    pix = rgb_im.getpixel((DATA[i][0] , DATA[i][1]))
    #print i
    #sheet.write(row,i,0)
    
    if pix==255:
        #print 1
        sheet.write(i,col,1)
    else:
        sheet.write(i,col,pix)
        #print pix

#print pix
wbk.save('data2.xls')
