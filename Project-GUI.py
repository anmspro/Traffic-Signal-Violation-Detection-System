from tkinter import *
from tkinter import messagebox
from PIL import Image, ImageTk
from tkinter import filedialog
import object_detection as od
import imageio
import cv2
import logging
import traceback
from utils.geometry import line_segment_intersection

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Window(Frame):
    def __init__(self, master=None):
        Frame.__init__(self, master)

        self.master = master
        self.pos = []
        self.line = []
        self.rect = []
        self.master.title("GUI")
        self.pack(fill=BOTH, expand=1)

        self.counter = 0

        menu = Menu(self.master)
        self.master.config(menu=menu)

        file = Menu(menu)
        file.add_command(label="Open", command=self.open_file)
        file.add_command(label="Exit", command=self.client_exit)
        menu.add_cascade(label="File", menu=file)
        
        analyze = Menu(menu)
        analyze.add_command(label="Region of Interest", command=self.regionOfInterest)
        menu.add_cascade(label="Analyze", menu=analyze)

        self.filename = "Images/home.jpg"
        self.imgSize = Image.open(self.filename)
        self.tkimage =  ImageTk.PhotoImage(self.imgSize)
        self.w, self.h = (1366, 768)
        
        self.canvas = Canvas(master = root, width = self.w, height = self.h)
        self.canvas.create_image(20, 20, image=self.tkimage, anchor='nw')
        self.canvas.pack()

    def open_file(self):
        """Open and load a video file for processing."""
        try:
            # Open file dialog with video file filter
            self.filename = filedialog.askopenfilename(
                title="Select Video File",
                filetypes=[
                    ("Video files", "*.mp4 *.avi *.mov *.mkv *.flv"),
                    ("All files", "*.*")
                ]
            )

            # User cancelled
            if not self.filename:
                return

            logger.info(f"Opening video file: {self.filename}")

            # Try to open video with OpenCV
            cap = cv2.VideoCapture(self.filename)
            if not cap.isOpened():
                raise ValueError("Unable to open video file. The file may be corrupted or in an unsupported format.")

            # Read first frame
            ret, image = cap.read()
            if not ret or image is None:
                raise ValueError("Unable to read video frames. The video may be empty or corrupted.")

            # Get FPS using imageio
            try:
                reader = imageio.get_reader(self.filename)
                fps = reader.get_meta_data()['fps']
                logger.info(f"Video FPS: {fps}")
            except Exception as e:
                logger.warning(f"Unable to read FPS metadata: {e}. Using default.")
                fps = 30  # Default FPS

            # Save preview image
            preview_path = 'G:/Traffic Violation Detection/Traffic Signal Violation Detection System/Images/preview.jpg'
            cv2.imwrite(preview_path, image)

            # Display preview
            self.show_image(preview_path)

            logger.info("Video loaded successfully")
            cap.release()

        except FileNotFoundError as e:
            error_msg = f"File not found: {self.filename}"
            logger.error(error_msg)
            messagebox.showerror("File Not Found", error_msg)

        except ValueError as e:
            error_msg = str(e)
            logger.error(error_msg)
            messagebox.showerror("Invalid Video", error_msg)

        except Exception as e:
            error_msg = f"Unexpected error opening video: {str(e)}"
            logger.error(f"{error_msg}\n{traceback.format_exc()}")
            messagebox.showerror("Error", error_msg)


    def show_image(self, frame):
        self.imgSize = Image.open(frame)
        self.tkimage =  ImageTk.PhotoImage(self.imgSize)
        self.w, self.h = (1366, 768)

        self.canvas.destroy()

        self.canvas = Canvas(master = root, width = self.w, height = self.h)
        self.canvas.create_image(0, 0, image=self.tkimage, anchor='nw')
        self.canvas.pack()

    def regionOfInterest(self):
        root.config(cursor="plus") 
        self.canvas.bind("<Button-1>", self.imgClick) 

    def client_exit(self):
        exit()

    def imgClick(self, event):

        if self.counter < 2:
            x = int(self.canvas.canvasx(event.x))
            y = int(self.canvas.canvasy(event.y))
            self.line.append((x, y))
            self.pos.append(self.canvas.create_line(x - 5, y, x + 5, y, fill="red", tags="crosshair"))
            self.pos.append(self.canvas.create_line(x, y - 5, x, y + 5, fill="red", tags="crosshair"))
            self.counter += 1

        # elif self.counter < 4:
        #     x = int(self.canvas.canvasx(event.x))
        #     y = int(self.canvas.canvasy(event.y))
        #     self.rect.append((x, y))
        #     self.pos.append(self.canvas.create_line(x - 5, y, x + 5, y, fill="red", tags="crosshair"))
        #     self.pos.append(self.canvas.create_line(x, y - 5, x, y + 5, fill="red", tags="crosshair"))
        #     self.counter += 1

        if self.counter == 2:
            #unbinding action with mouse-click
            self.canvas.unbind("<Button-1>")
            root.config(cursor="arrow")
            self.counter = 0

            #show created virtual line
            print(self.line)
            print(self.rect)
            img = cv2.imread('G:/Traffic Violation Detection/Traffic Signal Violation Detection System/Images/preview.jpg')
            cv2.line(img, self.line[0], self.line[1], (0, 255, 0), 3)
            cv2.imwrite('G:/Traffic Violation Detection/Traffic Signal Violation Detection System/Images/copy.jpg', img)
            self.show_image('G:/Traffic Violation Detection/Traffic Signal Violation Detection System/Images/copy.jpg')

            ## for demonstration
            # (rxmin, rymin) = self.rect[0]
            # (rxmax, rymax) = self.rect[1]

            # tf = False
            # tf |= self.intersection(self.line[0], self.line[1], (rxmin, rymin), (rxmin, rymax))
            # print(tf)
            # tf |= self.intersection(self.line[0], self.line[1], (rxmax, rymin), (rxmax, rymax))
            # print(tf)
            # tf |= self.intersection(self.line[0], self.line[1], (rxmin, rymin), (rxmax, rymin))
            # print(tf)
            # tf |= self.intersection(self.line[0], self.line[1], (rxmin, rymax), (rxmax, rymax))
            # print(tf)

            # cv2.line(img, self.line[0], self.line[1], (0, 255, 0), 3)

            # if tf:
            #     cv2.rectangle(img, (rxmin,rymin), (rxmax,rymax), (255,0,0), 3)
            # else:
            #     cv2.rectangle(img, (rxmin,rymin), (rxmax,rymax), (0,255,0), 3)

            # cv2.imshow('traffic violation', img)
            
            #image processing
            self.main_process()
            print("Executed Successfully!!!")

            #clearing things
            self.line.clear()
            self.rect.clear()
            for i in self.pos:
                self.canvas.delete(i)

    def main_process(self):
        """Process video and detect traffic violations."""
        cap = None
        writer = None

        try:
            video_src = self.filename

            if not video_src:
                raise ValueError("No video file selected")

            logger.info("Starting violation detection process...")

            # Open video capture
            cap = cv2.VideoCapture(video_src)
            if not cap.isOpened():
                raise ValueError("Unable to open video file for processing")

            # Get video metadata
            try:
                reader = imageio.get_reader(video_src)
                fps = reader.get_meta_data()['fps']
                logger.info(f"Processing at {fps} FPS")
            except Exception as e:
                logger.warning(f"Unable to read FPS, using default: {e}")
                fps = 30

            # Setup output writer
            output_path = 'G:/Traffic Violation Detection/Traffic Signal Violation Detection System/Resources/output/output.mp4'
            try:
                writer = imageio.get_writer(output_path, fps=fps)
            except Exception as e:
                raise IOError(f"Unable to create output video writer: {e}")

            j = 1
            while True:
                ret, image = cap.read()

                if not ret or image is None:
                    logger.info(f"Processed {j-1} frames successfully")
                    break

                image_h, image_w, _ = image.shape
                new_image = od.preprocess_input(image, od.net_h, od.net_w)

                # run the prediction
                yolos = od.yolov3.predict(new_image)
                boxes = []

                for i in range(len(yolos)):
                    # decode the output of the network
                    boxes += od.decode_netout(yolos[i][0], od.anchors[i], od.obj_thresh, od.nms_thresh, od.net_h, od.net_w)

                # correct the sizes of the bounding boxes
                od.correct_yolo_boxes(boxes, image_h, image_w, od.net_h, od.net_w)

                # suppress non-maximal boxes
                od.do_nms(boxes, od.nms_thresh)

                # draw bounding boxes on the image using labels
                image2 = od.draw_boxes(image, boxes, self.line, od.labels, od.obj_thresh, j)

                writer.append_data(image2)

                # cv2.imwrite('E:/Virtual Traffic Light Violation Detection System/Images/frame'+str(j)+'.jpg', image2)
                # self.show_image('E:/Virtual Traffic Light Violation Detection System/Images/frame'+str(j)+'.jpg')

                cv2.imshow('Traffic Violation', image2)

                logger.info(f"Processing frame {j}")

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    logger.info("Processing stopped by user")
                    break

                j = j+1

            logger.info("Processing completed successfully")

        except KeyboardInterrupt:
            logger.info("Processing interrupted by user")
            messagebox.showinfo("Interrupted", "Processing was interrupted")

        except Exception as e:
            error_msg = f"Error during video processing: {str(e)}"
            logger.error(f"{error_msg}\n{traceback.format_exc()}")
            messagebox.showerror("Processing Error", error_msg)

        finally:
            # Clean up resources
            if writer is not None:
                try:
                    writer.close()
                    logger.info("Output video writer closed")
                except Exception as e:
                    logger.error(f"Error closing video writer: {e}")

            if cap is not None:
                try:
                    cap.release()
                    logger.info("Video capture released")
                except Exception as e:
                    logger.error(f"Error releasing video capture: {e}")

            cv2.destroyAllWindows()

root = Tk()
app = Window(root)
root.geometry("%dx%d"%(535, 380))
root.title("Traffic Violation")

root.mainloop()