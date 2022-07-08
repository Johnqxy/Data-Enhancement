import cv2
import os
import glob
import random
import numpy as np
from tkinter import *
from tkinter import filedialog
import natsort
import json
import copy

class MixImg():
    def __init__(self):
        self.bg_dir = r""
        self.img_dir = r""
        self.save_dir = r""
        self.annotation_dir = r""
        self.detect_class = r""
        self.rename_dir = r""
        self.first_index = None
        self.file_tail = r""
        self.new_name = r""
        self.handle_flag = False
        self.predefined_classes = []
        self.objectList = []
        self.mix_flag = False
        self.rename_flag = False
        self.root_window = None

    def re_imsize(self, num, num1):
        if (num < num1):
            num1 = num * 0.30
        return num1

    def judge_rate(self, num, num1, rate):
        if (num / num1 != rate):
            num = num1 * rate
        return num, num1

    # 遍历xml里面每个object的值如果相同就不插入

    def rename(self):
        for file_name in natsort.natsorted(glob.glob(os.path.join(self.rename_dir, "*"))):
            # 设置旧文件名（就是路径+文件名）
            # print(file_name)
            oldname = file_name
            if (self.first_index == None):
                self.first_index = 0
            newname = self.rename_dir + '\\' + self.new_name + str(self.first_index) + self.file_tail
            # print(newname)
            # # 设置新文件名
            # newname = file_name.split('\\')[-1]+ str(n + 1) + '.jpg'
            #
            # # 用os模块中的rename方法对文件改名
            os.rename(oldname, newname)
            print(oldname, '======>', newname)

            self.first_index = int(self.first_index) + 1
        self.first_index = None

    def make_annotation(self, iw, igauss, irate, mixdir, n, label, mask, raw_point):
        IMAGES_LIST = os.listdir(mixdir)

        # print(IMAGES_LIST)
        for bg_filename in natsort.natsorted(IMAGES_LIST):
            n += 1
            point = copy.deepcopy(raw_point)
            if bg_filename.endswith(('.jpg', '.png', '.jpeg', '.bmp')):
                bg_filename = os.path.join(mixdir, bg_filename)
                relative_imgdir = bg_filename.split('/')[-1]
                save_filename = self.save_dir + '/'
                # save_annoname=cut_img_filename+"mix" + os.path.splitext(bg_filename)[0].split('\\')[-1]
                save_annoname = "mix" + str(n)
                save_filename += save_annoname
                bg = cv2.imread(bg_filename)
                bg_h, bg_w = bg.shape[:2]
                # img_h=self.re_imsize(bg_h,img_h)
                img_w = int(self.re_imsize(bg_w, iw))
                # img_h,img_w=self.judge_rate(img_h,img_w,hw_rate)
                img_h = int(img_w * irate)
                img_h = int(self.re_imsize(bg_h, img_h))
                img = cv2.resize(igauss, (img_w, img_h))
                mask = cv2.resize(mask, (img_w, img_h))

                rnd = [random.randint(0, bg_h - img_h), random.randint(0, bg_w - img_w)]

                object_information = [rnd[1], rnd[0], img.shape[1] + rnd[1], img.shape[0] + rnd[0],
                                      ]  # x1,y1,x2,y2,name
                for i in range(len(point)):
                    point[i][0] = point[i][0] + rnd[1]
                    point[i][1] = point[i][1] + rnd[0]
                # print(point)
                json_name = ('{}/{}.json'.format(self.annotation_dir, save_annoname))

                if (os.path.exists(json_name)):
                    json_dict = {'label': label, 'points': point,
                                 "group_id": NONE, 'shape_type': 'polygen', "flags": {}}
                    with open('{}/{}.json'.format(self.annotation_dir, save_annoname), 'w') as fs:
                        data = json.load(fs)
                        data['shapes'].append(json_dict)
                        fs.close()


                else:
                    json_dict = {'shapes': [{'label': label, 'points': point,
                                             "group_id": NONE, 'shape_type': 'polygen', "flags": {}}],
                                 "imagePath": (save_filename + '.jpg').split('/')[-1]
                        , "imageHeight": bg_h, "imageWidth": bg_w}
                    with open('{}/{}.json'.format(self.annotation_dir, save_annoname), 'w') as fs:
                        json.dump(json_dict, fs)
                        fs.close()

                # print(iname + "标签已保存")

                mask_inv = cv2.bitwise_not(mask)
                bg[object_information[1]:object_information[3],
                object_information[0]:object_information[2]] = cv2.bitwise_and(
                    bg[object_information[1]:object_information[3],
                    object_information[0]:object_information[2]], bg[object_information[1]:object_information[3],
                                                                  object_information[0]:object_information[2]],
                    mask=mask_inv)

                fg = cv2.bitwise_and(img, img, mask=mask)

                bg[object_information[1]:object_information[3],
                object_information[0]:object_information[2]] = cv2.add(bg[object_information[1]:object_information[3],
                                                                       object_information[0]:object_information[2]], fg)
                cv2.imwrite(save_filename + '.jpg', bg)
                print("图片已保存到" + save_filename)
                object_information.clear()
                point.clear()
        # self.handle_flag=True
        # print(img_filename+"标签已保存")

    def mix(self):
        x = []
        y = []
        relative_point = []
        for imgdir_filename in natsort.natsorted(glob.glob(os.path.join(self.img_dir, "*"))):
            imgdir_filename, type = imgdir_filename.split('.')
            if type == 'json':
                continue
            original_img = cv2.imread(imgdir_filename + '.' + type)
            with open(imgdir_filename + '.json', 'r') as fp:
                js = json.load(fp)
                for i in range(0, len(js['shapes'])):
                    points = js['shapes'][i]['points']
                    label = js['shapes'][i]['label']
                    for point in points:
                        x.append(point[0])
                        y.append(point[1])
                        max_x = int(max(x))
                        min_x = int(min(x))
                        max_y = int(max(y))
                        min_y = int(min(y))
                    for point in points:
                        relative_point.append([point[0] - min_x, point[1] - min_y])
                    bg_index = 0
                    min_rect_img = original_img[min_y:max_y, min_x:max_x]
                    mask = np.zeros(min_rect_img.shape, min_rect_img.dtype)
                    cv2.fillPoly(mask, [np.array(relative_point, dtype=np.int32)], (255, 255, 255))
                    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                    img_h, img_w = min_rect_img.shape[:2]
                    hw_rate = img_h / img_w
                    if (self.handle_flag):
                        self.make_annotation(img_w, min_rect_img, hw_rate, self.save_dir, bg_index, label, mask,
                                             relative_point)
                    else:
                        self.make_annotation(img_w, min_rect_img, hw_rate, self.bg_dir, bg_index, label, mask,
                                             relative_point)
                    self.handle_flag = True
                    x.clear()
                    y.clear()
                    relative_point.clear()

    def client(self):
        def creatWindow():
            self.root_window.destroy()
            window()

        def judge(str):
            if (str):
                text = "你已选择" + str
            else:
                text = "你还未选择文件夹，请选择"
            return text

        def test01():
            self.img_dir = r""
            self.img_dir += filedialog.askdirectory()
            creatWindow()

        def test02():
            self.bg_dir = r""
            self.bg_dir += filedialog.askdirectory()
            creatWindow()

        def test03():
            self.save_dir = r""
            self.save_dir += filedialog.askdirectory()
            self.annotation_dir = r""
            self.annotation_dir = self.save_dir
            creatWindow()

        def test04():
            self.annotation_dir = r""
            self.annotation_dir += filedialog.askdirectory()
            creatWindow()

        def test05():
            self.detect_class = r""
            self.detect_class += filedialog.askopenfilename()
            creatWindow()

        def test06():
            self.mix_flag = True
            self.mix()
            creatWindow()

        def test07():
            self.rename_dir = r""
            self.rename_dir += filedialog.askdirectory()
            creatWindow()

        def test08(t_n, t_f, t_t):
            self.new_name = t_n
            self.first_index = t_f
            self.file_tail = t_t
            self.rename_flag = True
            self.rename()
            creatWindow()

        def window():
            self.root_window = Tk()
            self.root_window.title("")
            screen_width = self.root_window.winfo_screenwidth()  # 获取显示区域的宽度
            screen_height = self.root_window.winfo_screenheight()  # 获取显示区域的高度
            tk_width = 800  # 设定窗口宽度
            tk_height = 800  # 设定窗口高度
            tk_left = int((screen_width - tk_width) / 2)
            tk_top = int((screen_height - tk_width) / 2)
            self.root_window.geometry('%dx%d+%d+%d' % (tk_width, tk_height, tk_left, tk_top))
            self.root_window.minsize(tk_width, tk_height)  # 最小尺寸
            self.root_window.maxsize(tk_width, tk_height)  # 最大尺寸
            self.root_window.resizable(width=False, height=False)
            btn_1 = Button(self.root_window, text='请选择你要标注的图片文件夹', command=test01,
                           height=0)
            btn_1.place(x=175, y=40, anchor='w')

            text = judge(self.img_dir)
            text_label = Label(self.root_window, text=text)
            text_label.place(x=166, y=70, anchor='w')

            btn_2 = Button(self.root_window, text='请选择你背景图的文件夹', command=test02,
                           height=0)
            btn_2.place(x=175, y=100, anchor='w')
            text = judge(self.bg_dir)
            text_label = Label(self.root_window, text=text)
            text_label.place(x=166, y=130, anchor='w')

            btn_3 = Button(self.root_window, text='请选择要保存图片的文件夹', command=test03,
                           height=0)
            btn_3.place(x=175, y=160, anchor='w')
            text = judge(self.save_dir)
            text_label = Label(self.root_window, text=text)
            text_label.place(x=166, y=190, anchor='w')

            btn_4 = Button(self.root_window, text='请选择你要保存的xml文件夹(.xml)', command=test04,
                           height=0)
            btn_4.place(x=175, y=220, anchor='w')
            text = judge(self.annotation_dir)
            text_label = Label(self.root_window, text=text)
            text_label.place(x=166, y=250, anchor='w')

            btn_5 = Button(self.root_window, text='请选择需要生成数据的类别文件(.txt)', command=test05,
                           height=0)
            btn_5.place(x=175, y=280, anchor='w')
            text = judge(self.detect_class)
            text_label = Label(self.root_window, text=text)
            text_label.place(x=166, y=310, anchor='w')

            btn_6 = Button(self.root_window, text='开始生成', command=test06,
                           height=0)
            btn_6.place(x=175, y=340, anchor='w')
            if (self.mix_flag):
                text = "生成完成"
            else:
                text = "等待生成"
            text_label = Label(self.root_window, text=text)
            text_label.place(x=166, y=370, anchor='w')

            btn_7 = Button(self.root_window, text='请选择你要改名的文件夹', command=test07,
                           height=0)
            btn_7.place(x=20, y=100, anchor='w')

            text_label = Label(self.root_window, text="改变后的名字")
            text_label.place(x=20, y=130, anchor='w')
            t_7_name = StringVar()
            t_7_name.set("mix")
            t_7_name = Entry(self.root_window, textvariable=t_7_name)
            t_7_name.place(x=100, y=130, width=50, anchor='w')

            text_label = Label(self.root_window, text="文件序号从何值开始")
            text_label.place(x=20, y=160, anchor='w')
            t_7_first = StringVar()
            t_7_first.set("0")
            t_7_first = Entry(self.root_window, textvariable=t_7_first)
            t_7_first.place(x=137, y=160, width=20, anchor='w')

            text_label = Label(self.root_window, text="文件名后缀")
            text_label.place(x=20, y=190, anchor='w')
            t_7_tail = StringVar()
            t_7_tail.set(".jpg")
            t_7_tail = Entry(self.root_window, textvariable=t_7_tail, width=20)
            t_7_tail.place(x=90, y=190, width=30, anchor='w')
            if (self.rename_dir):
                btn_8 = Button(self.root_window, text='开始改名(慎用)',
                               command=lambda: test08(t_7_name.get(), t_7_first.get(), t_7_tail.get()),
                               height=0)
                btn_8.place(x=40, y=220, anchor='w')

            if (self.rename_flag):
                text = "改名完成"
            else:
                text = "等待改名"
            text_label = Label(self.root_window, text=text)
            text_label.place(x=40, y=250, anchor='w')

            self.root_window.mainloop()

        window()


if __name__ == '__main__':
    mixImg = MixImg()
    mixImg.client()
