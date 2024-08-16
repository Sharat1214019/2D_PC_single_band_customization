import numpy as np
import matplotlib.pyplot as plt

#function for the dispersion relation from theory
def omega_theory_mono(qx, qy, nx, ny, kxy, knxy, m=1, a=1):
    omega_sq = 0
    for i in range(nx+1):
        for j in range(ny+1):
            if i*j==0:
                gamma = 1/2
            else:
                gamma = 1
                
            if i==0 and j==0:
                omega_sq += 0 
            else:
                omega_sq += gamma*(kxy[i][j]*(1 - np.cos(qx*i*a+qy*j*a)) + knxy[i][j]*(1 - np.cos(qx*i*a-qy*j*a)))
    omega = np.sqrt((2/m)*omega_sq)
    return omega

#function to create contour plot
def contour_plot_full(x, y, z, map_color='winter'):
    fig,ax = plt.subplots(figsize=(10,10))
    n_lines = np.linspace(0, 1, 15)
    
    ax.contour(x/np.pi, y/np.pi, z/np.max(z), levels=n_lines, colors='black', linestyles='solid', linewidths=2)
    cntr = ax.contourf(x/np.pi, y/np.pi, z/np.max(z), levels=250, cmap=map_color)
    
    ax.set_xticks([-1.0, -0.5, 0, 0.5, 1.0])
    ax.set_xticklabels(['', '', '', '', ''])
    ax.set_yticks([-1.0, -0.5, 0, 0.5, 1.0])
    ax.set_yticklabels(['', '', '', '', ''])
    
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.tick_params(direction='out', length=8, width=2, axis='both')
    ax.set_aspect('equal')
    #ax.set_axis_off()
    plt.show()
    return 

#functio to create 3D Surface plot for single band
def surface_plot_single_band(x, y, z, map_color):
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')#, proj_type='ortho')
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    ax.plot_surface(x/np.pi, y/np.pi, z/np.max(z), cmap=map_color, edgecolor='black', linewidth=0.2, rstride=25, cstride=25)
    ax.set_xticks([-1.0, 0, 1.0])
    ax.set_xticklabels(['', '', ''])
    ax.set_yticks([-1.0, 0, 1.0])
    ax.set_yticklabels(['', '', ''])
    ax.set_zticks([0, 1])
    ax.set_zticklabels(['', ''])
    ax.tick_params(direction='out', length=8, width=2, axis='x')
    ax.tick_params(direction='out', length=8, width=2, axis='y')
    ax.tick_params(direction='out', length=8, width=2, axis='z')
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([0, 1])
    #ax.set_axis_off()
    plt.show()
    return

grid_size = 1001
qx, qy = np.meshgrid(np.linspace(-np.pi, np.pi, grid_size), np.linspace(-np.pi, np.pi, grid_size))

# Case I, Figure - 2(a), (b)
target_omega = lambda x, y: np.sqrt(1/2) * np.sqrt(5-np.cos(x)-np.cos(y)-np.cos(2*x)-np.cos(2*y)-np.cos(1*x+3*y))
kxy = np.load('Case I - Fig 2(a),(b)/kxy_case_i.npy')
knxy = np.load('Case I - Fig 2(a),(b)/knxy_case_i.npy')
# Case II, Figure - 2(c),(d)
'''
target_omega = lambda x, y: ( - (abs(x) - np.pi)**2 - (abs(y) - np.pi)**2 + 2*np.pi**2 )
kxy = np.load('Case II - Fig 2(c),(d)/kxy_case_ii.npy')
knxy = np.load('Case II - Fig 2(c),(d)/knxy_case_ii.npy')
'''
# Case III, Figure - 2(e),(f)
'''
target_omega = 'Case III - Fig 2(e),(f)/target_case_iii.txt'
kxy = np.load('Case III - Fig 2(e),(f)/kxy_case_iii.npy')
knxy = np.load('Case III - Fig 2(e),(f)/knxy_case_iii.npy')
'''
# Case IV, Figure - 2(g), (h)
'''
target_omega = 'Case IV - Fig 2(g),(h)/target_case_iv.txt'
kxy = np.load('Case IV - Fig 2(g),(h)/kxy_case_iv.npy')
knxy = np.load('Case IV - Fig 2(g),(h)/knxy_case_iv.npy')
'''
# Case VI, Figure - 6(c), (d)
'''
target_omega = 'Case VI - Fig 6(c),(d)/target_case_vi.txt'
kxy = np.load('Case VI - Fig 6(c),(d)/kxy_case_vi.npy')
knxy = np.load('Case VI - Fig 6(c),(d)/knxy_case_vi.npy')
'''  
# Case VII, Figure - 6(e), (f)
'''
target_omega = lambda x, y: np.sqrt(2-np.cos(np.sqrt(x**2 + y**2))-np.cos(2 * np.sqrt(x**2 + y**2)))
kxy = np.load('Case VII - Fig 6(e),(f)/kxy_case_vii.npy')
knxy = np.load('Case VII - Fig 6(e),(f)/knxy_case_vii.npy')
'''
# Case VIII, Figure - 6(g), (h)
'''
target_omega = lambda x, y: (x**2 + y**2 < (np.pi/2)**2)*np.sqrt(x**2 + y**2) + (x**2 + y**2 >= (np.pi/2)**2)*np.pi/2
kxy = np.load('Case VIII - Fig 6(g),(h)/kxy_case_viii.npy')
knxy = np.load('Case VIII - Fig 6(g),(h)/knxy_case_viii.npy')
'''
# Case IX, Figure - 8(a), (b)
'''
target_omega = np.load('Case IX - Fig 8(a),(b)/target_case_ix.npy')
kxy = np.load('Case IX - Fig 8(a),(b)/kxy_case_ix.npy')
knxy = np.load('Case IX - Fig 8(a),(b)/knxy_case_ix.npy')
'''
# Case X, Figure - 8(c), (d)
'''
target_omega = np.load('Case X - Fig 8(c),(d)/target_case_x.npy')
kxy = np.load('Case X - Fig 8(c),(d)/kxy_case_x.npy')
knxy = np.load('Case X - Fig 8(c),(d)/knxy_case_x.npy')
'''
nx, ny = kxy.shape[0] - 1, kxy.shape[1] - 1
achieved_omega = lambda x, y: omega_theory_mono(x, y, nx, ny, kxy, knxy)

# For analytical target dispersion - Case I, II, VII, VIII
contour_plot_full(qx, qy, target_omega(qx,qy), 'coolwarm')
surface_plot_single_band(qx, qy, target_omega(qx,qy),'coolwarm')
contour_plot_full(qx, qy, achieved_omega(qx,qy),'coolwarm')
surface_plot_single_band(qx, qy, (achieved_omega(qx,qy)),'coolwarm')

# For numerical target dispersion - Case I, II, VII, VIII
'''
contour_plot_full(qx, qy, target_omega, 'coolwarm')
surface_plot_single_band(qx, qy, target_omega,'coolwarm')
contour_plot_full(qx, qy, achieved_omega(qx,qy),'coolwarm')
surface_plot_single_band(qx, qy, (achieved_omega(qx,qy)),'coolwarm')
'''