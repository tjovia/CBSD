from PIL import Image
import xlwt
# Create workbook and worksheet
wbk = xlwt.Workbook()
sheet = wbk.add_sheet('testData')

im = Image.open("2..png") #Can be many different formats.
pix = im.load()
print im.size #Get the width and hight of the image for iterating over

x,y=im.size
rgb_im = im.convert('RGB')
row = 0
for i in range(0,x):
    for j in range(0,y):
        col=0
        r, g, b = rgb_im.getpixel((i , j))
        if r!=255 and g!=255 and b!=255:
            sheet.write(row,col,i)
            col=col+1
            sheet.write(row,col,j)
            col=col+1
            sheet.write(row,col,r)
            col=col+1
            sheet.write(row,col,g)
            col=col+1
            sheet.write(row,col,b)
            row=row+1 
            #print row, col, i, j
# include the other coulurs ie. blue en green

wbk.save('allTestDatanew.xls')
