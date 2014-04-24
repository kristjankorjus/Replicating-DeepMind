import numpy as np


def NTSC_To_Grayscale():
    f=open("ColorsRaw")
    lines=f.readlines()
    f.close()

    #print len(lines)
    Rs=[]
    Gs=[]
    Bs=[]

    count=0
    for line in lines:
        if line[:3]=="RGB":
            count+=1
            print line[6:12]
            hexa=line[6:12]
            r=int(hexa[:2], 16)
            g=int(hexa[2:4], 16)
            b=int(hexa[4:], 16)
            print r, g, b
            Rs.append(r)
            Gs.append(g)
            Bs.append(b)

    print "count", count

    Rs=np.reshape(Rs, (8, 16))
    Gs=np.reshape(Gs, (8, 16))
    Bs=np.reshape(Bs, (8, 16))


    #f=open("R.dat", "w")
    #txt="\n".join(str(elem)[1:-1] for elem in Rs)
    #f.write(txt)
    #f.close()
    #f=open("B.dat", "w")
    #txt="\n".join(str(elem)[1:-1] for elem in Bs)
    #f.write(txt)
    #f.close()
    #f=open("G.dat", "w")
    #txt="\n".join(str(elem)[1:-1] for elem in Gs)
    #f.write(txt)
    #f.close()

    Grayscale=0.21*Rs+0.71*Gs+0.07*Bs #based on luminoscity
    #print 0.21*Rs
    #print 0.71*Gs
    #print 0.07*Gs
    
    #print np.shape(Rs)
    #print np.shape(Bs)
    #print np.shape(Gs)
    #print np.shape(Grayscale)
    np.savetxt("Grayscale.dat", Grayscale)  #saves the matrix into a file
    #np.savetxt("R.dat", Rs)
    return Grayscale


#########################CALLING the function##########################

matr1=NTSC_To_Grayscale() #you can either take the returned grayscale matrix
#or read it from the file where it was saved
matr=np.loadtxt("Grayscale.dat") #this is how you can read the saved file
#print matr
#print matr1
#print matr1==matr

