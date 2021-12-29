import tkinter as tk

def my_linfit(x, y):

    upper = 0
    downer = 0
    x_avg = average_num(x)
    y_avg = average_num(y)
    for i in range(len(x)):
        upper += x[i]*(y[i]-y_avg)
        downer += x[i]*(x[i]-x_avg)
    a = upper/downer
    b = y_avg - a*x_avg
    return a, b

def average_num(x):
    result_avg = 0
    for i in range(len(x)):
        result_avg += x[i]
    result_avg = result_avg/len(x)
    return result_avg


def main():
    dot_r = 0.5
    x_list = []
    y_list = []
    window = tk.Tk()
    canvas = tk.Canvas(window, width=200, height=200, background ='white')
    canvas.grid(row=0, column=0)
    # 2 following lines indicate which way coordinate will go for easy check
    # Still use tkinter coordinate, not cut point of 2 following lines
    canvas.create_line(10, 10, 10, 200, arrow=tk.LAST)
    canvas.create_line(10, 10, 200, 10, arrow=tk.LAST)
    def left_click(event):
        x = event.x
        y = event.y
        x_list.append(x)
        y_list.append(y)
        # Still use the original coordinate of tkinter
        print("Your coordinate: " + str(x) + ", " + str(y))
        canvas.create_oval(x-dot_r, y-dot_r, x+dot_r, y+dot_r)
    window.bind("<Button-1>", left_click)

    def right_click(event):
        a, b = my_linfit(x_list, y_list)
        canvas.create_line(0, b, 200, 200*a+b)
        print("Your a is " +str(a)+" and your b is "+str(b))
        window.unbind("<Button-1>")

    window.bind("<Button-3>", right_click)


    window.mainloop()
if __name__ == "__main__":
    main()


