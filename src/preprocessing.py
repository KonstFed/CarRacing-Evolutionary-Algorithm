import numpy as np
import cv2


class FrameParser():

    def __init__(self, mat='image',
                 l_u_corner=np.array([65, 45]),
                 r_d_corner=np.array([76, 50]),
                 ui_rotation_corners=np.array([[86, 37], [91, 58]]),
                 ui_speed_corners=np.array([[84, 13], [93, 13]])) -> None:
        # all coordinates are stored in format [y, x]
        self.l_u_corner = l_u_corner
        self.r_d_corner = r_d_corner
        self.mat = mat
        self.road_rgb = [105, 105, 105]

        self.s_p_forward1 = np.array(
            [l_u_corner[0], (l_u_corner[1] + r_d_corner[1])//2])
        self.s_p_forward2 = self.s_p_forward1 + [0, 1]

        self.s_p_left_side = np.array(
            [(l_u_corner[0] + r_d_corner[0]) // 2, l_u_corner[1]])
        self.s_p_rigt_side = np.array(
            [(l_u_corner[0] + r_d_corner[0]) // 2, r_d_corner[1]])

        self.s_p_left_angled = l_u_corner
        self.s_p_right_angled = np.array([l_u_corner[0], r_d_corner[1]])

        self.ui_rotation_corners = ui_rotation_corners
        self.ui_speed_coreners = ui_speed_corners

    def carCenter(self):
        center = (self.left_upper_corner + self.right_down_corner)//2
        return center

    def _ray(self, binary: np.ndarray, delta: np.array, start_pos: np.array, debug=False):
        if debug:
            out = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        count = 0
        pos = np.copy(start_pos)
        while (pos > [0, 0]).all() and (pos < binary.shape[:2]).all():
            if debug:
                out[pos[0], pos[1]] = (0, 0, 255)
                cv2.imshow("debug", out)
            if binary[pos[0], pos[1]] == 0:
                return count
            count += 1
            pos += delta

        return count

    def _getRays(self, binary):
        """
        Ray is distance to end of the road.
        Returns 5 rays: forward, left side, right side, left 45*, right 45*.
        """
        forward1 = self._ray(binary, np.array(
            [-1, 0]), self.s_p_forward1)
        forward2 = self._ray(binary, np.array([-1, 0]), self.s_p_forward2)
        forward = min(forward1, forward2)

        l_side = self._ray(binary, np.array([0, -1]), self.s_p_left_side)
        r_side = self._ray(binary, np.array([0, 1]), self.s_p_rigt_side)

        l_angled = self._ray(binary, np.array(
            [-1, -1]), self.s_p_left_angled) * np.sqrt(2)
        r_angled = self._ray(binary, np.array(
            [-1, 1]), self.s_p_right_angled) * np.sqrt(2)

        return [forward, l_side, r_side, l_angled, r_angled]

    def _binarizeWorld(self, frame):
        img = np.abs(frame - self.road_rgb)
        img = img.astype(np.uint8)
        out = np.zeros(img.shape[:2]).astype(np.uint8)
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if img[i, j].sum() < 25:
                    out[i, j] = 255
                else:
                    out[i, j] = 0
        return out

    def _getRotation(self, frame):
        ui_rotation_frame = frame[self.ui_rotation_corners[0][0]: self.ui_rotation_corners[1]
                                  [0] + 1, self.ui_rotation_corners[0][1]:self.ui_rotation_corners[1][1] + 1]
        c_x = (self.ui_rotation_corners[0][1] + self.ui_rotation_corners[1]
               [1]) // 2 - self.ui_rotation_corners[0][1]

        binary = np.zeros(ui_rotation_frame.shape[:2])
        for y in range(ui_rotation_frame.shape[0]):
            for x in range(ui_rotation_frame.shape[1]):
                if ui_rotation_frame[y, x][1] > 40:
                    binary[y, x] = 255

        c_y = binary.shape[0] // 2
        count = 0
        for x in range(c_x+1, binary.shape[1]):
            if binary[c_y, x] == 0:
                break
            count += 1

        if count == 0:
            for x in range(c_x, -1, -1):
                if binary[c_y, x] == 0:
                    break
                count -= 1
        return count

    def _getSpeed(self, frame):
        ui_frame = frame[self.ui_speed_coreners[0][0]:self.ui_speed_coreners[1]
                         [0]+1, self.ui_speed_coreners[0][1]:self.ui_speed_coreners[1][1]+1]
        speed = 0
        for i in range(ui_frame.shape[0]-1, -1, -1):
            if ui_frame[i, 0][0] == 0:
                break
            speed += ui_frame[i, 0][0] / 255
        return speed

    def process(self, frame: np.array):
        """
        Process input image to input for GA.
        Returns array consisting of speed, rotation, forward ray, left side ray, right side ray, left 45 degree ray, right 45 degree ray.
        """
        world_frame = frame[:85,]  # remove bottom control panel
        binary = self._binarizeWorld(world_frame)
        rays = self._getRays(binary)
        rotation = self._getRotation(frame)
        speed = self._getSpeed(frame)
        return np.array([speed, rotation] + rays)

    def save(self, frame: np.array, filename='screen.png'):
        cv2.imwrite(filename, frame)
