import cv2
import numpy as np
import keras
from Soduku_solver import solve, myGrid, myarray2str

# for testing
# import time
# current_milli_time = lambda: int(round(time.time() * 1000))

# import model
json_file = open("Sources/model.json")
loaded_model_json = json_file.read()
json_file.close()
loaded_model1 = keras.models.model_from_json(loaded_model_json)
loaded_model1.load_weights("Sources/model.h5")

# set up camera
cap = cv2.VideoCapture(0)
winWidth = 450
winHeight = 450
cap.set(3, winWidth)  # set width
cap.set(4, winHeight)  # set height
cap.set(10, 100)  # brightness


def get_rect(imgthresh):  # outputs the corner coords of biggest rectangle in thresh image
    maxarea = 0
    biggestpoly = np.array([])
    contours, hierarchy = cv2.findContours(imgthresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 2000:
            peri = cv2.arcLength(cnt, True)
            poly = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if area > maxarea and len(poly) == 4:
                biggestpoly = poly
                maxarea = area
    return biggestpoly


def get_reorder(poly):  # reorders the corner coordinates of polygon to match warp
    poly = poly.reshape((4, 2))
    polynew = np.zeros((4, 1, 2), np.int32)
    add = poly.sum(1)
    polynew[0] = poly[np.argmin(add)]
    polynew[3] = poly[np.argmax(add)]

    diff = np.diff(poly, axis=1)
    polynew[1] = poly[np.argmin(diff)]
    polynew[2] = poly[np.argmax(diff)]

    return polynew


def get_warp(image, poly):
    pts1 = np.float32(get_reorder(poly))  # from original img
    pts2 = np.float32([[0, 0], [winWidth, 0], [0, winHeight], [winWidth, winHeight]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgoutput = cv2.warpPerspective(image, matrix, (winWidth, winHeight))
    return imgoutput


def get_projection(warped, poly, image):
    pts1 = np.float32(get_reorder(poly))  # points in the original img
    pts2 = np.float32([[0, 0], [winWidth, 0], [0, winHeight], [winWidth, winHeight]])
    matrix = cv2.getPerspectiveTransform(pts2, pts1)
    unwarped = cv2.warpPerspective(warped, matrix, (640, 480))
    # put in right order
    pts1 = pts1.astype('int32')
    idx = [3, 1, 0, 2]
    pts_new = pts1[idx]
    # cover old grid in black
    image = cv2.fillPoly(image, [pts_new], (0, 0, 0))
    out = image + unwarped
    return out


class Sudoku:
    def __init__(self):
        self.failed_grids = np.empty([0, 9, 9])
        self.success_grids = np.empty([0, 9, 9])
        self.prev_sol = np.empty([0, 9, 9])
        self.grid = np.empty([9, 9])
        self.gridSol = np.empty([9, 9])

    def getdigit(self, image):
        img_norm = image / 255
        img_resized = cv2.resize(img_norm, (28, 28))
        img_reshaped = img_resized.reshape(1, 28, 28, 1)
        return np.argmax(loaded_model1.predict(img_reshaped), axis=-1)

    def getgrid(self, th):
        sudoku = cv2.resize(th, (450, 450))
        gridisplay = 255 - sudoku.copy()
        grid1 = np.zeros([9, 9])
        boarder = 4

        for i in range(9):
            for j in range(9):
                # drawing outer digit rectangle
                margin = 0
                topleft = (j * 50 + boarder + margin, i * 50 + boarder + margin)
                bottomright = ((j + 1) * 50 - boarder - margin, (i + 1) * 50 - boarder - margin)
                gridisplay = cv2.rectangle(gridisplay, topleft, bottomright, (0, 0, 255), 1)

                digit = sudoku[i * 50 + boarder:(i + 1) * 50 - boarder, j * 50 + boarder:(j + 1) * 50 - boarder]

                # crop for center of box
                margin = 8
                crop_dig = digit[margin:-margin, margin:-margin]

                # drawing center rectangle
                topleft = (j * 50 + boarder + margin, i * 50 + boarder + margin)
                bottomright = ((j + 1) * 50 - boarder - margin, (i + 1) * 50 - boarder - margin)
                gridisplay = cv2.rectangle(gridisplay, topleft, bottomright, (0, 0, 255), 1)

                if crop_dig.sum() > 15000:  # if center region of square is not empty
                    contours, _ = cv2.findContours(digit, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
                    for cnt in contours:
                        x, y, w, h = cv2.boundingRect(cnt)
                        a_ratio = float(w) / h
                        size = w * h
                        if 0.9 >= a_ratio >= 0.2 and size > 200:
                            numberimg = digit[y:y + h, x:x + w]
                            top, bottom, left, right = [10, 10, 20, 20]
                            numberimg_boarder = cv2.copyMakeBorder(numberimg, top, bottom, left, right,
                                                                   cv2.BORDER_CONSTANT, value=(0, 0, 0))
                            grid1[i][j] = self.getdigit(numberimg_boarder)
                else:
                    grid1[i][j] = 0

            # if the first row is the same as the first row of a solved grid
            # assume it is the same grid
            for subgrid in self.success_grids:
                if np.all(grid1[0] == subgrid[0]):
                    grid1 = subgrid
                    break

        grid = grid1.astype(int)
        self.grid = grid
        return grid, gridisplay

    def in_old_failed_grid(self):
        for subgrid in self.failed_grids:
            if np.all(self.grid == subgrid):
                return True
        return False

    def in_old_success_grid(self):
        for subgrid in self.success_grids:
            if np.all(self.grid == subgrid):
                return True, subgrid
        return False, []

    def solve(self):
        # see if grid has been successfully solved before
        success, success_grid = self.in_old_success_grid()
        if success:
            self.gridSol = success_grid
        # see if grid has failed to be solved before
        if self.in_old_failed_grid():
            # use prev solution
            self.gridSol = self.prev_sol
        # if new grid
        else:
            # solve grid
            string_of_num = myarray2str(self.grid)
            self.gridSol = np.array(myGrid(solve(string_of_num)))

    def show_solution(self, image, warp_image):
        # if there is a solution
        if self.gridSol.size > 0:
            self.prev_sol = self.gridSol
            self.success_grids = np.append(self.success_grids, [self.grid], 0)

            # find missing numbers
            copy = warp_image.copy()
            for i in range(9):
                for j in range(9):
                    if self.grid[i][j] == 0:
                        # and draw solution digit
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        lowleft = (48 * j + 20, 48 * i + 40)
                        fontscale = 1
                        color = (255, 0, 0)
                        cv2.putText(copy, str(int(self.gridSol[i][j])), lowleft, font, fontscale, color, 2)
            # cv2.imshow('filled puzzle', copy)
            proj = get_projection(copy, get_reorder(sqrPoly), image)
            cv2.imshow('projection', proj)

        else:  # if no solution
            self.failed_grids = np.append(self.failed_grids, [self.grid], 0)
            # project img without drawing
            copy = warpImg.copy()
            proj = get_projection(copy, get_reorder(sqrPoly), image)
            cv2.imshow('projection', proj)


if __name__ == '__main__':
    Sudoku = Sudoku()
    while True:
        # for testing
        # spot1 = current_milli_time()
        # print('begin "{}"'.format(spot1))

        # Read image
        _, img = cap.read()
        imgContour = img.copy()
        projection = img
        cv2.imshow('projection', projection)

        # Process image
        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clean = cv2.fastNlMeansDenoising(imgGray)
        threshFine = cv2.adaptiveThreshold(clean, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 13, 2)
        threshFine = 255 - threshFine
        # cv2.imshow('thresh', threshFine)

        # draw bounding points
        sqrPoly = get_rect(threshFine)
        cv2.drawContours(imgContour, sqrPoly, -1, (0, 0, 255), 20)
        # cv2.imshow('contour points', imgContour)

        # ROI
        if sqrPoly.size > 0:  # if there is a square poly region
            # warp poly into square
            warpSqr = get_warp(threshFine, get_reorder(sqrPoly))
            warpImg = get_warp(img, get_reorder(sqrPoly))  # for drawing

            # extract grid
            thegrid, gridwboxes = Sudoku.getgrid(warpSqr)
            cv2.imshow('testing', gridwboxes)

            # solve grid
            Sudoku.solve()

            # show solution
            Sudoku.show_solution(img, warpImg)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
