import numpy as np

# points
Mouth = [[48, 49], [49, 50], [50, 51], [51, 52], [52, 53], [53, 54], [54, 55], [55, 56], [56, 57], \
         [57, 58], [58, 59], [59, 48], [60, 61], [61, 62], [62, 63], [63, 64], [64, 65], [65, 66], \
         [66, 67], [67, 60]]

Nose = [[27, 28], [28, 29], [29, 30], [30, 31], [30, 35], [31, 32], [32, 33], \
        [33, 34], [34, 35], [27, 31], [27, 35]]#,[31,35],[32,34],[32,30],[30,34],[32,29],[34,29],[31,29],[29,35],[30,33]]

leftBrow = [[17, 18], [18, 19], [19, 20], [20, 21]]#,[18,20],[21,19]]
rightBrow = [[22, 23], [23, 24], [24, 25], [25, 26]]#,[23,25],[22,24]]

leftEye = [[36, 37], [37, 38], [38, 39], [39, 40], [40, 41], [36, 41]]
rightEye = [[42, 43], [43, 44], [44, 45], [45, 46], [46, 47], [42, 47]]

other = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], \
         [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], \
         [12, 13], [13, 14], [14, 15], [15, 16]]

Lips =  [[60, 59],[59, 67],[67, 58],[58, 66],[66, 57],[57, 65],[65, 56],[56,64],[64,55],[49,53]]
Lips2 =  [[55, 65],[56, 66],[58, 60]]

Lips3 =  [[60, 49],[49, 61],[61, 50],[50, 62],[62, 51],[51, 63],[63, 52],[52,64],[64,53],[59,55],[58,56]]
Lips4 =  [[62, 53],[50, 62],[60, 50]]


cavity = [[60, 64],[61, 67],[62, 66],[63, 65],[61, 65],[63, 67],[61,64],[64,67],[60,63],[60,65]]
eyeball_left = [[37, 41],[38, 40],[37, 40],[38, 41],[36,38],[36,40],[37,39],[39,41]] #[36, 39]
eyeball_right = [[43, 47],[44, 46],[43, 46],[44, 47],[42,44],[42,46],[43,45],[45,47]] #[42, 45]

eyelid = [37,38 ,43,44]
#faceLmarkLookup = Mouth + Nose + leftBrow + rightBrow + leftEye + rightEye + other #+ other2
faceLmarkLookup = [] #Mouth# + leftEye + rightEye# + Lips + Lips2 + Lips3 + Lips4
faceLmarkLookup2 = [] #leftEye + rightEye #eyeball_left + eyeball_right# + cavity

faceLmarkLookup3 = leftBrow + rightBrow# + other + Nose 
faceLmarkLookup4 = [] #Nose

import matplotlib.pyplot as plt
plt.switch_backend('agg') 
import matplotlib as mpl
mpl.use('Agg')

from tqdm import tqdm
 
def eyelid_animation(frames,time=15,low=8,high=30,framerate=25):
    
    count_animation = int(np.ceil(frames.shape[0]/framerate/time))-1 #округляем до целого меньшего кол-во морганий за ролик

    if count_animation > 0:
        speed_animation = np.random.randint(low=low, high=high, size=count_animation)
        for i in range(0,count_animation): #округляем до целого меньшего кол-во морганий за ролик
            frame_start_delay = int(np.random.randint(low=1, high=time, size=1)[0]) #в какую секунду с начала отрезка начинаем анимацию моргания
            
            frame_start = i * time * framerate + frame_start_delay * framerate
            
            speed_eyelid_animation = np.linspace(frames[frame_start][37][1], frames[frame_start+speed_animation[i]][42][1], speed_animation[i])
            
            y_a = 0
            
            for y_animation in speed_eyelid_animation:
                print("animation edit" ,y_animation , i  )
                frames[frame_start+y_a][37][1] = y_animation
                frames[frame_start+y_a][38][1] = y_animation
                
                frames[frame_start+y_a][43][1] = y_animation
                frames[frame_start+y_a][44][1] = y_animation
                y_a += 1
            
            frames[frame_start+y_a+1][37][1] = frames[frame_start+speed_animation[i]][42][1]
            frames[frame_start+y_a+1][38][1] = frames[frame_start+speed_animation[i]][42][1]
                
            frames[frame_start+y_a+1][43][1] = frames[frame_start+speed_animation[i]][42][1]
            frames[frame_start+y_a+1][44][1] = frames[frame_start+speed_animation[i]][42][1]            

            for y_animation in speed_eyelid_animation[::-1]:
                print("animation edit" ,y_animation , i  )
                frames[frame_start-y_a][37][1] = y_animation
                frames[frame_start-y_a][38][1] = y_animation
                
                frames[frame_start-y_a][43][1] = y_animation
                frames[frame_start-y_a][44][1] = y_animation
                y_a -= 1
            
    return frames

def mark_video(fake_lmark,img_video_dir):

    data = fake_lmark
    
    
    fake_lmark = np.reshape(data, (data.shape[1], 68, 2))
    
    #нормализация лица по первому кадру
    fake_lmark[1:,0:48,:] = fake_lmark[0,0:48,:]
    #только моргание
    #fake_lmark[1:,:,:] = fake_lmark[0,:,:]
    #fake_lmark[:,69,:] = 0 
    
    
    frames = fake_lmark#[:,305:307,:]
    
    if len(frames.shape) < 3:
        frames = np.reshape(frames, (frames.shape[0], frames.shape[1]/2, 2))
    
    frames = eyelid_animation(frames,10,8,30,25)
    
    
    lookup = faceLmarkLookup
    lookup2 = faceLmarkLookup2
    lookup3 = faceLmarkLookup3
    lookup4 = faceLmarkLookup4
                
    for i in tqdm(range(frames.shape[0])):            
                fig = plt.figure(figsize=(10, 10))
    
                xLim = [-1, 1]
                yLim = [-1, 1]            
                plt.xlim(xLim)
                plt.ylim(yLim)
                plt.axis('off')
       
                plt.gca().invert_yaxis()
                  
                ax = plt.gca()
    
                eye1 = plt.Circle(((frames[i,37, 0] + frames[i,40, 0])/2, (frames[i,37, 1] + frames[i,40, 1])/2), 0.06, facecolor='forestgreen',edgecolor='darkgreen', zorder=15)
                eye1_ = plt.Circle(((frames[i,37, 0] + frames[i,40, 0])/2, (frames[i,37, 1] + frames[i,40, 1])/2), 0.02, facecolor='black',edgecolor='dimgray', zorder=15)
                eye2 = plt.Circle(((frames[i,43, 0] + frames[i,46, 0])/2, (frames[i,37, 1] + frames[i,46, 1])/2), 0.06, facecolor='forestgreen',edgecolor='darkgreen', zorder=15)
                eye2_ = plt.Circle(((frames[i,43, 0] + frames[i,46, 0])/2, (frames[i,37, 1] + frames[i,46, 1])/2), 0.02, facecolor='black',edgecolor='dimgray', zorder=15)
                
                
                ax.add_artist(eye1)
                ax.add_artist(eye1_)            
                ax.add_artist(eye2)
                ax.add_artist(eye2_)                
    
                
    
                
                lines = [plt.plot([], [], 'r', linewidth=3, zorder=15)[0] for _ in range(3*len(lookup))]
                l, = plt.plot([], [], 'ko', marker='_', mfc='red',
                              mec='red', ms=1, mew=1)
              
                cnt = 0
                for refpts in lookup:
 
                    lines[cnt].set_data([frames[i,refpts[1], 0], frames[i,refpts[0], 0]], [frames[i, refpts[1], 1], frames[i,refpts[0], 1]])
                    cnt+=1
                   
                    
                lines2 = [plt.plot([], [], 'b', zorder=15)[0] for _ in range(3*len(lookup2))]
                l2, = plt.plot([], [], 'ko', marker='_', mfc='black',
                              mec='black', ms=2, mew=2)
                     
                cnt = 0
                for refpts in lookup2:

                    lines2[cnt].set_data([frames[i,refpts[1], 0], frames[i,refpts[0], 0]], [frames[i, refpts[1], 1], frames[i,refpts[0], 1]])
                    cnt+=1
    
    
                lines3 = [plt.plot([], [], 'b', linewidth=4, zorder=20)[0] for _ in range(3*len(lookup3))]
                l3, = plt.plot([], [], 'ko', ms=4)
              
                cnt = 0
                
                for refpts in lookup3:
                    
                    lines3[cnt].set_data([frames[i,refpts[1], 0], frames[i,refpts[0], 0]], [frames[i, refpts[1], 1], frames[i,refpts[0], 1]])
                    cnt+=1
                                
                lines4 = [plt.plot([], [], 'k', linewidth=1, zorder=15)[0] for _ in range(3*len(lookup3))]
                l4, = plt.plot([], [], 'ko', ms=4)
               
                cnt = 0
                
                
                for refpts in lookup4:
                    
                    lines4[cnt].set_data([frames[i,refpts[1], 0], frames[i,refpts[0], 0]], [frames[i, refpts[1], 1], frames[i,refpts[0], 1]])
                    cnt+=1
    
              
                frames[i,0, 0] = frames[i,0, 0] + 0.0
                frames[i,0, 1] = frames[i,0, 1] - 0.35
    
                frames[i,16, 0] = frames[i,16, 0] - 0.0
                frames[i,16, 1] = frames[i,16, 1] - 0.35
                
                frames[i,17, 0] = frames[i,24, 0] - 0.0
                frames[i,17, 1] = frames[i,24, 1] - 0.25
                
                frames[i,18, 0] = frames[i,19, 0] - 0.0
                frames[i,18, 1] = frames[i,19, 1] - 0.25
                
                plt.fill(frames[i,0:19, 0], frames[i,0:19, 1], facecolor='peru', edgecolor='chocolate', linewidth=1,zorder=10)                  
                
                
                
                frames[i,28, 0] = frames[i,31, 0]
                frames[i,28, 1] = frames[i,31, 1]           
    
                frames[i,29, 0] = frames[i,30, 0]
                frames[i,29, 1] = frames[i,30, 1]   
    
                
                #нос
                plt.fill(frames[i,27:31, 0], frames[i,27:31, 1], facecolor='sandybrown', edgecolor='saddlebrown', linewidth=1,zorder=21)
                plt.fill(frames[i,30:36, 0], frames[i,30:36, 1], facecolor='sienna', edgecolor='saddlebrown', linewidth=1,zorder=21)
                plt.fill(frames[i,[30,27,35], 0], frames[i,[30,27,35], 1], facecolor='peru', edgecolor='saddlebrown', linewidth=1,zorder=21)
              
                
                #губы
                frames[i,59, 0] = frames[i,59, 0]# - 0.02
                frames[i,59, 1] = frames[i,59, 1]# - 0.02              
                            
                plt.fill(frames[i,[48,49,50,51,52,53,54,64,63,62,61,60], 0], frames[i,[48,49,50,51,52,53,54,64,63,62,61,60], 1], facecolor='salmon', edgecolor='saddlebrown', linewidth=1,zorder=30) 
                plt.fill(frames[i,[54,55,56,57,58,59,48,60,67,66,65,64], 0], frames[i,[54,55,56,57,58,59,48,60,67,66,65,64], 1], facecolor='salmon', edgecolor='saddlebrown', linewidth=1,zorder=30) 
                
                
                plt.fill(frames[i,36:42, 0], frames[i,36:42, 1], facecolor='whitesmoke', edgecolor='saddlebrown', linewidth=1,zorder=14)
                plt.fill(frames[i,42:48, 0], frames[i,42:48, 1], facecolor='whitesmoke', edgecolor='saddlebrown', linewidth=1,zorder=14)
    
    
                #рот
                plt.fill(frames[i,60:68, 0], frames[i,60:68, 1], facecolor='maroon', edgecolor='crimson', linewidth=1,zorder=18)
                
                #веки
                plt.fill(frames[i,[19,36,37,38,39,20], 0], frames[i,[19,36,37,38,39,20], 1], facecolor='peru', linewidth=1,zorder=17)
                plt.fill(frames[i,[23,42,43,44,45,24], 0], frames[i,[23,42,43,44,45,24], 1], facecolor='peru', linewidth=1,zorder=17)
                
                plt.fill(frames[i,[2,36,41,40,39,4], 0], frames[i,[2,36,41,40,39,4], 1], facecolor='peru', linewidth=1,zorder=17)
                plt.fill(frames[i,[12,42,47,46,45,14], 0], frames[i,[12,42,47,46,45,14], 1], facecolor='peru', linewidth=1,zorder=17)            
                
                
                #зубы
                frames[i,7, 1] -= 0.44
                frames[i,9, 1] -= 0.44  
                frames[i,7, 0] -= 0.02
                frames[i,9, 0] += 0.02  
                frames[i,8, 1] = frames[i,57, 1] - 0.01
                plt.fill(frames[i,[8,7,9], 0], frames[i,[8,7,9], 1], facecolor='lightyellow' ,edgecolor='palegoldenrod', linewidth=1,zorder=19)                        
                plt.fill(frames[i,[4,6,10,12,54,55,56,57,58,59,48], 0], frames[i,[4,6,10,12,54,55,56,57,58,59,48], 1], facecolor='peru', linewidth=1,zorder=20)            
                
                
                frames[i,32, 1] += 0.22
                frames[i,34, 1] += 0.22  
                frames[i,32, 0] = frames[i,49, 0] - 0.01
                frames[i,34, 0] = frames[i,53, 0] + 0.01           
                
                plt.fill(frames[i,[32,34,30], 0], frames[i,[32,34,30], 1], facecolor='lightyellow' ,edgecolor='palegoldenrod', linewidth=1,zorder=19)                        
                plt.fill(frames[i,[48,49,50,51,52,53,54,11,15,1,6], 0], frames[i,[48,49,50,51,52,53,54,11,15,1,6], 1], facecolor='peru', linewidth=1,zorder=20)            
                plt.savefig(img_video_dir + str(i) + '.png', transparent=True)