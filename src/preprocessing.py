import numpy as np
import cv2


class FrameParser:
    def __init__(
        self,
        road_rgb=[103, 103, 103],
        ui_angle_corners=np.array([[86, 37], [91, 58]]),
        ui_speed_corners=np.array([[84, 13], [93, 13]]),
        ui_rotation_center=np.array([88, 71]),
    ):
        self.road_rgb = road_rgb
        self.ui_angle_corners = ui_angle_corners
        self.ui_speed_corners = ui_speed_corners
        self.ui_rotation_center = ui_rotation_center
        self.ui_rotation_delta = 13

    def _binarizeWorld(self, frame):
        """Binarize image by distance to road color.

        Due to simplicity of input image manhattan distance less than 10 from road rgb works perfectly for this task. Road pixel is 255, others are 0.
        Args:
            frame (np.array): BGR np.array of a frame without UI

        Returns:
            np.array: Binarized image, road pixel is 255 others pixels are 0
        """
        img = np.abs(frame - self.road_rgb)
        img = img.astype(np.uint8)
        out = np.zeros(img.shape[:2]).astype(np.uint8)
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if img[i, j].sum() < 10:
                    out[i, j] = 255
                else:
                    out[i, j] = 0
        return out

    def _getRotation(self, frame):
        """Extract gyroscope rotation from UI

        Args:
            frame (np.array): BGR representation of frame

        Returns:
            float: rotation [-1, 1]
        """
        pos = np.copy(self.ui_rotation_center)
        l_count = 0
        while l_count <= self.ui_rotation_delta:
            if frame[pos[0], pos[1]][0] < 40:
                break
            l_count += 1
            pos += [0, -1]

        r_count = 0
        pos = np.copy(self.ui_rotation_center) + [0, 1]
        while r_count <= self.ui_rotation_delta:
            if frame[pos[0], pos[1]][0] < 40:
                break
            pos += [0, 1]

        if l_count > r_count:
            return -l_count / self.ui_rotation_delta
        else:
            return r_count / self.ui_rotation_delta

    def _getWheelAngle(self, frame):
        """Parse UI to get angle of wheels.

        Args:
            frame (np.array): RGB np.array of image.

        Returns:
            float: angle [-1, 1].
        """
        ui_rotation_frame = frame[
            self.ui_angle_corners[0][0] : self.ui_angle_corners[1][0] + 1,
            self.ui_angle_corners[0][1] : self.ui_angle_corners[1][1] + 1,
        ]
        c_x = (
            self.ui_angle_corners[0][1] + self.ui_angle_corners[1][1]
        ) // 2 - self.ui_angle_corners[0][1]

        binary = np.zeros(ui_rotation_frame.shape[:2])
        for y in range(ui_rotation_frame.shape[0]):
            for x in range(ui_rotation_frame.shape[1]):
                if ui_rotation_frame[y, x][1] > 40:
                    binary[y, x] = 255

        c_y = binary.shape[0] // 2
        count = 0
        for x in range(c_x + 1, binary.shape[1]):
            if binary[c_y, x] == 0:
                break
            count += 1

        if count == 0:
            for x in range(c_x, -1, -1):
                if binary[c_y, x] == 0:
                    break
                count -= 1
        return count / 10

    def _getSpeed(self, frame):
        """Parse UI to get speed.

        Args:
            frame (np.array): BGR np.array of image

        Returns:
            float: speed [0, 1]
        """
        ui_frame = frame[
            self.ui_speed_corners[0][0] : self.ui_speed_corners[1][0] + 1,
            self.ui_speed_corners[0][1] : self.ui_speed_corners[1][1] + 1,
        ]
        speed = 0
        for i in range(ui_frame.shape[0] - 1, -1, -1):
            if ui_frame[i, 0][0] == 0:
                break
            speed += ui_frame[i, 0][0] / 255
        return speed / (self.ui_speed_corners[1][0] - self.ui_speed_corners[0][0] + 1)

    def save(self, frame: np.array, filename="screen.png"):
        """Save frame to filename path.

        Args:
            frame (np.array): frame
            filename (str, optional): path and name. Defaults to 'screen.png'.
        """
        cv2.imwrite(filename, frame)

    def process(self, frame: np.array):
        """Abstract method. Will process input image to input for GA."""
        raise NotImplementedError()


class BinaryFrameParser(FrameParser):
    def __init__(self, crop_factor = 2) -> None:
        super().__init__()
        self.crop_factor = crop_factor

    def process(self, frame: np.array):
        """Returns binarized image w/o UI

        Args:
            frame (np.array): input frame BGR

        Returns:
            np.array: 85 x 96 binary image where road is 1, else 0
        """
        binary = self._binarizeWorld(frame)
        size = binary.shape[0] # 96 = 48 * 2 = 24 * 4
        binary = binary[size//4:, size - size//4] # 48 x 48
        binary = cv2.resize(binary, (24, 24)) # 24, 24
        return binary / 255 


class RayFrameParser(FrameParser):
    def __init__(
        self,
        l_u_corner=np.array([66, 46]),
        r_d_corner=np.array([74, 49]),
        ui_angle_corners=np.array([[86, 37], [91, 58]]),
        ui_speed_corners=np.array([[84, 13], [93, 13]]),
        ui_rotation_center=np.array([88, 71]),
        road_rgb=[103, 103, 103],
    ):
        """Given class parse frames of Gymnasium CarRacing game using function process
        Args:
            l_u_corner (np.array, optional): pixel coordinates of left upper corner of car. Defaults to np.array([65, 45]).
            r_d_corner (np.array, optional): pixel coordinates of right down corner of car. Defaults to np.array([76, 50]).
            ui_rotation_corners (np.array, optional): left up and right down corners of rotation bar on UI. Defaults to np.array([[86, 37], [91, 58]]).
            ui_speed_corners (np.array, optional): left up and right down corners of speed bar on UI. Defaults to np.array([[84, 13], [93, 13]]).
        """
        super().__init__(
            road_rgb=road_rgb,
            ui_angle_corners=ui_angle_corners,
            ui_speed_corners=ui_speed_corners,
            ui_rotation_center=ui_rotation_center,
        )
        # all coordinates are stored in format [y, x]
        self.l_u_corner = l_u_corner
        self.r_d_corner = r_d_corner

        self.s_p_forward1 = np.array(
            [l_u_corner[0], (l_u_corner[1] + r_d_corner[1]) // 2]
        )
        self.s_p_forward2 = self.s_p_forward1 + [0, 1]

        self.s_p_left_side = np.array(
            [(l_u_corner[0] + r_d_corner[0]) // 2, l_u_corner[1]]
        )
        self.s_p_rigt_side = np.array(
            [(l_u_corner[0] + r_d_corner[0]) // 2, r_d_corner[1]]
        )

        self.s_p_left_angled = l_u_corner
        self.s_p_right_angled = np.array([l_u_corner[0], r_d_corner[1]])

    def _ray(
        self,
        binary: np.ndarray,
        delta: np.array,
        start_pos: np.array,
        until: int,
        debug=True,
    ):
        """Compute distance by ray-casting from start_pos along delta.

        Distance computed by iterativly adding delta to start_pos. Then checking if it's road and add to count.
        If end of the image or not road met return count.
        Args:
            binary (np.ndarray): binary representation of frame, where road is 255 and everything else 0
            delta (np.array): 2d array that will be added to start_pos to check. To work properly values should be from [-1, 1]
            start_pos (np.array): from where we start ray casting
            until (int): send ray until we see until pixel. 0 for grass, 255 for road
            debug (bool, optional): If given true will show ray. Defaults to False.

        Returns:
            int: distance
        """
        if debug:
            out = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        count = 0
        pos = np.copy(start_pos) + delta
        while (pos > [0, 0]).all() and (pos < binary.shape[:2]).all():
            if debug:
                out[pos[0], pos[1]] = (0, 0, 255)
                # cv2.imshow("debug", out)
            if binary[pos[0], pos[1]] == until:
                return count
            count += 1
            pos += delta
        return count

    def _getRays(self, binary, until):
        """Compute forward, left side, right side, left 45, right 45 rays.
        Due to even size of image forward ray is computed as minimal of two rays that lies around center of car.

        Args:
            binary (np.array): binary representation of image where road is 255 and everything else 0
            until (int): send rays until we see until pixel. 0 for grass, 255 for road
        Returns:
            List[float]: 5 int element array, each in [0, 1]
        """
        forward1 = self._ray(binary, np.array([-1, 0]), self.s_p_forward1, until)
        forward2 = self._ray(binary, np.array([-1, 0]), self.s_p_forward2, until)
        forward = min(forward1, forward2)
        max_forward = self.s_p_forward1[0] - 1
        forward = forward / max_forward

        max_side = self.s_p_left_side[1] - 1
        l_side = (
            self._ray(binary, np.array([0, -1]), self.s_p_left_side, until) / max_side
        )
        r_side = (
            self._ray(binary, np.array([0, 1]), self.s_p_rigt_side, until) / max_side
        )

        max_angled = min(self.s_p_left_angled[0], self.s_p_left_angled[1])
        l_angled = (
            self._ray(binary, np.array([-1, -1]), self.s_p_left_angled, until)
            / max_angled
        )
        r_angled = (
            self._ray(binary, np.array([-1, 1]), self.s_p_right_angled, until)
            / max_angled
        )

        return [forward, l_side, r_side, l_angled, r_angled]

    def process(self, frame: np.array):
        """Process input image to input for GA.


        Args:
            frame (np.array): observation frame 96 x 96 RGB image.

        Returns:
            List[int]: in format of [speed, angle, rotation, forward ray grass, left side ray grass, right side ray grass, 45 left ray grass, 45 right ray grass, forward ray road, left side ray road, right side ray road, 45 left ray road, 45 right ray road]
        """
        world_frame = frame[:85,]  # remove bottom control panel
        binary = self._binarizeWorld(world_frame)
        rays_to_gras = self._getRays(binary, 0)
        rays_to_road = self._getRays(binary, 255)
        angle = self._getWheelAngle(frame)
        rotation = self._getRotation(frame)
        speed = self._getSpeed(frame)
        return np.array([speed, angle, rotation] + rays_to_gras + rays_to_road)
