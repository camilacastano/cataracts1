"""This is the first test on segmentation based on our dataset
called "pre-dataset"

Segmentation of cataract fragments using a simple thresholding method where we will
enclose cataract fragments in bounding boxes with individual IDs during phacoemulsification

For each fragment:
* Persistent and unique ID across video frames
* Colored bounding box with ID label

Dependencies: OpenCV, NumPy, PyTorch (for potential future use of deep learning models), scipy,
filterpy, rich, tqdm
"""

import os
import sys
import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict


"""Can detect regions that will correspond to other fragments
and asign them persistent ID?"""


video_path="/Users/camilacastano/Desktop/cataracts1/1/Maria Edith Feria Escalera OI .mp4"
output_vid="/Users/camilacastano/Desktop/cataracts1/1/Maria Edith Feria Escalera OI_output.mp4"


# filtering objects by their pixel size
min_area=80
max_area=50000
max_distance=70
max_missing=15

#self: object created

@dataclass
class Fragment:
    id: int
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    centroid: Tuple[int, int]  # (cx, cy)
    area: float
    missing: int = 0

class FragmentTracker:
    def __init__(self):
        self.next_id=0
        self.fragments: Dict[int, Fragment]={}

    def register(
            self,
            centroid,
            bbox,
            area
    ):
        self.fragments[self.next_id]=Fragment(
            id=self.next_id,
            centroid=centroid,
            bbox=bbox,
            area=area
        )

        self.next_id+=1

    def deregister(
            self,
            fragment_id
    ):
        
        if fragment_id in self.fragments:
            del self.fragments[fragment_id]

    def update(
            self,
            detections
    ):
        if len(detections)==0:
            remove_ids=[]

            for fid in self.fragments:
                self.fragments[fid].missing +=1
                if self.fragments[fid].missing>max_missing:
                    remove_ids.append(fid)

            for fid in remove_ids:
                self.deregister(fid)

        if len(self.fragments)==0:

            for det in detections:
                self.register(
                    det["centroid"],
                    det["bbox"],
                    det["area"]
                )
            return self.fragments
        
        fragment_ids=list(self.fragments.keys())

        used_fragments = set()
        used_detections=set()

        for det_idx, det in enumerate(detections):

            best_distance=np.inf
            best_id=None

            for fid in fragment_ids:
                if fid in used_fragments:
                    continue

                old_centroid=np.array(
                    self.fragments[fid].centroid
                )

                new_centroid=np.array(
                    det["centroid"]
                )


                distance=np.linalg.norm(
                    old_centroid-new_centroid
                )

                if distance<best_distance:
                    best_distance=distance
                    best_id=fid

            if (
                best_id is not None
                and best_distance<max_distance
            ):
                
                self.fragments[best_id].centroid=det["centroid"]
                self.fragments[best_id].bbox=det["bbox"]
                self.fragments[best_id].area=det["area"]
                self.fragments[best_id].missing=0

                used_fragments.add(best_id)
                used_detections.add(det_idx)

        for det_idx, det in enumerate(detections):
            if det_idx not in used_detections:
                self.register(
                    det["centroid"],
                    det["bbox"],
                    det["area"]
                )

        for fid in fragment_ids:
            if fid not in used_fragments:
                self.fragments[fid].missing+=1

        remove_ids=[]

        for fid in fragment_ids:
            if fid not in used_fragments:
                self.fragments[fid].missing+=1

        remove_ids=[]

        for fid in self.fragments:
            if self.fragments[fid].missing>max_missing:
                remove_ids.append(fid)

        for fid in remove_ids:
            self.deregister(fid)

        return self.fragments

def detect_fragments(frame):
    gray=cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    gray = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    mask=cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        21,
        5
    )

    kernel=np.ones((3,3),np.uint8)

    mask=cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    mask=cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    contours,_=cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    detections=[]

    for contour in contours:
        area=cv2.contourArea( 
            contour
        )

        if area<min_area:
            continue

        if area>max_area:
            continue

        x,y,w,h=cv2.boundingRect(
            contour
        )

        cx=x+w//2
        cy=y+h//2

        detections.append(
            {
                "centroid": (cx,cy),
                "bbox": (x,y,w,h),
                "area":area,
                "contour":contour
            }
        )

    return detections,mask


def get_color(id_):
    rng=np.random.RandomState(id_)
    return (
        int(rng.randint(50,255)),
        int(rng.randint(50,255)),
        int(rng.randint(50,255))
    )


def main():
    cap=cv2.VideoCapture(
        video_path
    )

    if not cap.isOpened():
        print("not found")
        return
    
    width=int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height=int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    fps=cap.get(cv2.CAP_PROP_FPS)

    writer=cv2.VideoWriter(
        output_vid,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (width,height)
    )

    tracker=FragmentTracker()
    frame_count=0

    while True:
        ret, frame=cap.read()

        if not ret:
            break

        frame_count +=1

        detections, mask = detect_fragments(frame)
        tracked=tracker.update(detections)

        for fid, fragment in tracked.items():
            color=get_color(fid)
            x,y,w,h = fragment.bbox

            cv2.rectangle(
                frame,
                (x,y),
                (x+w,y+h),
                color,
                2
            )

            cv2.circle(
                frame,
                fragment.centroid,
                4,
                color,
                -1
            )

            cv2.putText(
                frame,
                f"ID {fid}",
                (x,y-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

        cv2.putText(
            frame,
            f"Frame: {frame_count}",
            (20,30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255,255,255),
            2
        )

        writer.write(frame)
        
        cv2.imshow(
            "Tracking",
            frame
        )

        cv2.imshow(
            "Segmentation Mask",
            mask
        )

        key=cv2.waitKey(1)
        if key == 27:
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print("done")


if __name__=="__main__":
    main()




