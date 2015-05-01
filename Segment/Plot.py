''' Plot the downloaded ECG data '''

''' Library '''
import matplotlib.pyplot as plt
import scipy.io

''' Function '''


mat = scipy.io.loadmat('Data/100_file.mat')
plot_ex = mat['val'][0]
plot_ex2 = mat['val'][1]
time = [x / float(360) for x in range(len(plot_ex))]

print plot_ex2

# print len(time)
plt.figure()
plt.plot(time, plot_ex, 'b')
plt.plot(time, plot_ex2, 'r')
plt.grid()
plt.show()