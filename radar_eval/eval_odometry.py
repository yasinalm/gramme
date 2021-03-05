# Based on https://raw.githubusercontent.com/JiawangBian/SC-SfMLearner-Release/master/kitti_eval/kitti_odometry.py

import copy
import matplotlib as mpl
mpl.use('Agg') # No x-server
from matplotlib import pyplot as plt
import numpy as np
import os
from pathlib import Path
from glob import glob
import seaborn as sn

#sns.set(style=\"whitegrid\", rc={\"font.size\":8,\"axes.titlesize\":8,\"axes.labelsize\":5})
sn.set(style="whitegrid", font_scale=1.5)
sn.set_palette("bright", n_colors=4, color_codes=True)

class EvalOdom():
    """Evaluate odometry result
    Usage example:
        vo_eval = EvalOdom()
        vo_eval.eval(gt_pose, pred_pose, result_pose_txt_dir)
    """

    def __init__(self):
        self.lengths = [100, 200, 300, 400, 500, 600, 700, 800]
        self.num_lengths = len(self.lengths)
    

    def trajectory_distances(self, poses):
        """Compute distance for each pose w.r.t frame-0
        Args:
            poses (dict): {idx: 4x4 array}
        Returns:
            dist (float list): distance of each pose w.r.t frame-0
        """
        dist = [0]
        sort_frame_idx = sorted(poses.keys())
        for i in range(len(sort_frame_idx)-1):
            cur_frame_idx = sort_frame_idx[i]
            next_frame_idx = sort_frame_idx[i+1]
            P1 = poses[cur_frame_idx]
            P2 = poses[next_frame_idx]
            dx = P1[0, 3] - P2[0, 3]
            dy = P1[1, 3] - P2[1, 3]
            dz = P1[2, 3] - P2[2, 3]
            dist.append(dist[i]+np.sqrt(dx**2+dy**2+dz**2))
        return dist

    def rotation_error(self, pose_error):
        """Compute rotation error
        Args:
            pose_error (4x4 array): relative pose error
        Returns:
            rot_error (float): rotation error
        """
        a = pose_error[0, 0]
        b = pose_error[1, 1]
        c = pose_error[2, 2]
        d = 0.5*(a+b+c-1.0)
        rot_error = np.arccos(max(min(d, 1.0), -1.0))
        return rot_error

    def translation_error(self, pose_error):
        """Compute translation error
        Args:
            pose_error (4x4 array): relative pose error
        Returns:
            trans_error (float): translation error
        """
        dx = pose_error[0, 3]
        dy = pose_error[1, 3]
        dz = pose_error[2, 3]
        trans_error = np.sqrt(dx**2+dy**2+dz**2)
        return trans_error

    def last_frame_from_segment_length(self, dist, first_frame, length):
        """Find frame (index) that away from the first_frame with
        the required distance
        Args:
            dist (float list): distance of each pose w.r.t frame-0
            first_frame (int): start-frame index
            length (float): required distance
        Returns:
            i (int) / -1: end-frame index. if not found return -1
        """
        for i in range(first_frame, len(dist), 1):
            if dist[i] > (dist[first_frame] + length):
                return i
        return -1

    def calc_sequence_errors(self, poses_gt, poses_result):
        """calculate sequence error
        Args:
            poses_gt (dict): {idx: 4x4 array}, ground truth poses
            poses_result (dict): {idx: 4x4 array}, predicted poses
        Returns:
            err (list list): [first_frame, rotation error, translation error, length, speed]
                - first_frame: frist frame index
                - rotation error: rotation error per length
                - translation error: translation error per length
                - length: evaluation trajectory length
                - speed: car speed (#FIXME: 10FPS is assumed)
        """
        err = []
        dist = self.trajectory_distances(poses_gt)
        self.step_size = 10

        for first_frame in range(0, len(poses_gt), self.step_size):
            for i in range(self.num_lengths):
                len_ = self.lengths[i]
                last_frame = self.last_frame_from_segment_length(
                    dist, first_frame, len_
                )

                # Continue if sequence not long enough
                if last_frame == -1 or \
                        not(last_frame in poses_result.keys()) or \
                        not(first_frame in poses_result.keys()):
                    continue

                # compute rotational and translational errors
                pose_delta_gt = np.dot(
                    np.linalg.inv(poses_gt[first_frame]),
                    poses_gt[last_frame]
                )
                pose_delta_result = np.dot(
                    np.linalg.inv(poses_result[first_frame]),
                    poses_result[last_frame]
                )
                pose_error = np.dot(
                    np.linalg.inv(pose_delta_result),
                    pose_delta_gt
                )

                r_err = self.rotation_error(pose_error)
                t_err = self.translation_error(pose_error)

                # compute speed
                num_frames = last_frame - first_frame + 1.0
                speed = len_/(0.1*num_frames)

                err.append([first_frame, r_err/len_, t_err/len_, len_, speed])
        return err

    def save_sequence_errors(self, err, file_name):
        """Save sequence error
        Args:
            err (list list): error information
            file_name (str): txt file for writing errors
        """
        fp = open(file_name, 'w')
        for i in err:
            line_to_write = " ".join([str(j) for j in i])
            fp.writelines(line_to_write+"\n")
        fp.close()

    def compute_overall_err(self, seq_err):
        """Compute average translation & rotation errors
        Args:
            seq_err (list list): [[r_err, t_err],[r_err, t_err],...]
                - r_err (float): rotation error
                - t_err (float): translation error
        Returns:
            ave_t_err (float): average translation error
            ave_r_err (float): average rotation error
        """
        t_err = 0
        r_err = 0

        seq_len = len(seq_err)

        if seq_len > 0:
            for item in seq_err:
                r_err += item[1]
                t_err += item[2]
            ave_t_err = t_err / seq_len
            ave_r_err = r_err / seq_len
            return ave_t_err, ave_r_err
        else:
            return 0, 0

    def plot_trajectory(self, poses_gt, poses_result, result_dir):
        """Plot trajectory for both GT and prediction
        Args:
            poses_gt (dict): {idx: 4x4 array}; ground truth poses
            poses_result (dict): {idx: 4x4 array}; predicted poses
            seq (int): sequence index.
        """
        plot_keys = ["Ground Truth", "Ours"]
        fontsize_ = 20

        poses_dict = {}
        poses_dict["Ground Truth"] = poses_gt
        poses_dict["Ours"] = poses_result

        fig = plt.figure()
        ax = plt.gca()
        ax.set_aspect('equal')

        for key in plot_keys:
            pos_xz = []
            frame_idx_list = sorted(poses_dict["Ours"].keys())
            for frame_idx in frame_idx_list:
                # pose = np.linalg.inv(poses_dict[key][frame_idx_list[0]]) @ poses_dict[key][frame_idx]
                pose = poses_dict[key][frame_idx]
                pos_xz.append([pose[0, 3],  pose[1, 3]])
            pos_xz = np.asarray(pos_xz)
            plt.plot(pos_xz[:, 0],  pos_xz[:, 1], label=key)

        plt.legend(loc="upper right", prop={'size': fontsize_})
        plt.xticks(fontsize=fontsize_)
        plt.yticks(fontsize=fontsize_)
        plt.xlabel('x (m)', fontsize=fontsize_)
        plt.ylabel('y (m)', fontsize=fontsize_)
        fig.set_size_inches(10, 10)
        fig_pdf = result_dir/"trajectory.pdf"
        fig_png = result_dir/"trajectory.png"
        plt.savefig(fig_pdf, bbox_inches='tight', pad_inches=0)
        plt.savefig(fig_png, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

    # def plot_trajectory(self, pred, gt):
    #     gt_xyz = gt[:,:3,3]
    #     pred_xyz = pred[:,:3,3]
    #     fig, ax = plt.subplots(figsize=(8,8))
    #     sn.lineplot(x=pred_xyz[:,0], y=pred_xyz[:,1], sort=False, ax=ax, label='Ours')
    #     sn.lineplot(x=gt_xyz[:,0], y=gt_xyz[:,1], sort=False, ax=ax, label='Ground Truth')

    #     ax.set(xlabel='X (m)', ylabel='Y (m)')

    #     # Save fig
    #     plt.tight_layout()
    #     plt.savefig(str(Path(self.plot_path_dir)/'ro_pred_with_gt.pdf'), bbox_inches = 'tight', pad_inches = 0)
    #     plt.savefig(str(Path(self.plot_path_dir)/'ro_pred_with_gt.png'), bbox_inches = 'tight', pad_inches = 0)
    #     plt.close(fig)

    def plot_error(self, avg_segment_errs, result_dir):
        """Plot per-length error
        Args:
            avg_segment_errs (dict): {100:[avg_t_err, avg_r_err],...}
            seq (int): sequence index.
        """
        # Translation error
        plot_y = []
        plot_x = []
        for len_ in self.lengths:
            plot_x.append(len_)
            if len(avg_segment_errs[len_]) > 0:
                plot_y.append(avg_segment_errs[len_][0] * 100)
            else:
                plot_y.append(0)
        fontsize_ = 10
        fig = plt.figure()
        plt.plot(plot_x, plot_y, "bs-", label="Translation Error")
        plt.ylabel('Translation Error (%)', fontsize=fontsize_)
        plt.xlabel('Sequence Length (m)', fontsize=fontsize_)
        plt.legend(loc="upper right", prop={'size': fontsize_})
        fig.set_size_inches(5, 5)
        fig_pdf = result_dir/"trans_err.pdf"
        plt.savefig(fig_pdf, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

        # Rotation error
        plot_y = []
        plot_x = []
        for len_ in self.lengths:
            plot_x.append(len_)
            if len(avg_segment_errs[len_]) > 0:
                plot_y.append(avg_segment_errs[len_][1] / np.pi * 180 * 100)
            else:
                plot_y.append(0)
        fontsize_ = 10
        fig = plt.figure()
        plt.plot(plot_x, plot_y, "bs-", label="Rotation Error")
        plt.ylabel('Rotation Error (deg/100m)', fontsize=fontsize_)
        plt.xlabel('Sequence Length (m)', fontsize=fontsize_)
        plt.legend(loc="upper right", prop={'size': fontsize_})
        fig.set_size_inches(5, 5)
        fig_pdf = result_dir/"rot_err.pdf"
        plt.savefig(fig_pdf, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

    def compute_segment_error(self, seq_errs):
        """This function calculates average errors for different segment.
        Args:
            seq_errs (list list): list of errs; [first_frame, rotation error, translation error, length, speed]
                - first_frame: frist frame index
                - rotation error: rotation error per length
                - translation error: translation error per length
                - length: evaluation trajectory length
                - speed: car speed (#FIXME: 10FPS is assumed)
        Returns:
            avg_segment_errs (dict): {100:[avg_t_err, avg_r_err],...}    
        """

        segment_errs = {}
        avg_segment_errs = {}
        for len_ in self.lengths:
            segment_errs[len_] = []

        # Get errors
        for err in seq_errs:
            len_ = err[3]
            t_err = err[2]
            r_err = err[1]
            segment_errs[len_].append([t_err, r_err])

        # Compute average
        for len_ in self.lengths:
            if segment_errs[len_] != []:
                avg_t_err = np.mean(np.asarray(segment_errs[len_])[:, 0])
                avg_r_err = np.mean(np.asarray(segment_errs[len_])[:, 1])
                avg_segment_errs[len_] = [avg_t_err, avg_r_err]
            else:
                avg_segment_errs[len_] = []
        return avg_segment_errs

    def compute_ATE(self, gt, pred):
        """Compute RMSE of ATE
        Args:
            gt (4x4 array dict): ground-truth poses
            pred (4x4 array dict): predicted poses
        """
        errors = []
        idx_0 = list(pred.keys())[0]
        gt_0 = gt[idx_0]
        pred_0 = pred[idx_0]

        for i in pred:
            # cur_gt = np.linalg.inv(gt_0) @ gt[i]
            cur_gt = gt[i]
            gt_xyz = cur_gt[:3, 3]

            # cur_pred = np.linalg.inv(pred_0) @ pred[i]
            cur_pred = pred[i]
            pred_xyz = cur_pred[:3, 3]

            align_err = gt_xyz - pred_xyz

            # print('i: ', i)
            # print("gt: ", gt_xyz)
            # print("pred: ", pred_xyz)
            # input("debug")
            errors.append(np.sqrt(np.sum(align_err ** 2)))
        ate = np.sqrt(np.mean(np.asarray(errors) ** 2))
        return ate

    def compute_RPE(self, gt, pred):
        """Compute RPE
        Args:
            gt (4x4 array dict): ground-truth poses
            pred (4x4 array dict): predicted poses
        Returns:
            rpe_trans
            rpe_rot
        """
        trans_errors = []
        rot_errors = []
        for i in list(pred.keys())[:-1]:
            gt1 = gt[i]
            gt2 = gt[i+1]
            gt_rel = np.linalg.inv(gt1) @ gt2

            pred1 = pred[i]
            pred2 = pred[i+1]
            pred_rel = np.linalg.inv(pred1) @ pred2
            rel_err = np.linalg.inv(gt_rel) @ pred_rel

            trans_errors.append(self.translation_error(rel_err))
            rot_errors.append(self.rotation_error(rel_err))
        # rpe_trans = np.sqrt(np.mean(np.asarray(trans_errors) ** 2))
        # rpe_rot = np.sqrt(np.mean(np.asarray(rot_errors) ** 2))
        rpe_trans = np.mean(np.asarray(trans_errors))
        rpe_rot = np.mean(np.asarray(rot_errors))
        return rpe_trans, rpe_rot

   
    def write_result(self, f, seq, errs):
        """Write result into a txt file
        Args:
            f (IOWrapper)
            seq (int): sequence number
            errs (list): [ave_t_err, ave_r_err, ate, rpe_trans, rpe_rot]
        """
        ave_t_err, ave_r_err, ate, rpe_trans, rpe_rot = errs
        lines = []
        lines.append("Sequence: \t {} \n".format(seq))
        lines.append("Trans. err. (%): \t {:.3f} \n".format(ave_t_err*100))
        lines.append("Rot. err. (deg/100m): \t {:.3f} \n".format(ave_r_err/np.pi*180*100))
        lines.append("ATE (m): \t {:.3f} \n".format(ate))
        lines.append("RPE (m): \t {:.3f} \n".format(rpe_trans))
        lines.append("RPE (deg): \t {:.3f} \n\n".format(rpe_rot * 180 / np.pi))
        for line in lines:
            f.writelines(line)

    def eval(self, pred, gt, result_dir):
        """Evaluate the precited trajectory in terms of several metrics and save plots and stats to `result_dir`.

        Args:
            pred (numpy.ndarray): Predicted aligned trajectory in [N,4,4] format.
            gt (numpy.ndarray): Ground truth trajectory in [N,4,4] format.
            result_dir (pathlib.Path): Directory to save results.
        """

        poses_result = dict(zip(range(pred.shape[0]), pred))
        poses_gt = dict(zip(range(gt.shape[0]), gt))
       
        # Initialization
        ave_t_errs = []
        ave_r_errs = []
        seq_ate = []
        seq_rpe_trans = []
        seq_rpe_rot = []

        # Create result directory
        # error_dir = result_dir + "/errors"
        # self.plot_path_dir = result_dir + "/plot_path"
        # self.plot_error_dir = result_dir + "/plot_error"
        # result_txt = os.path.join(result_dir, "result.txt")
        result_txt = result_dir/"result.txt"
        f = open(result_txt, 'w')

        # if not os.path.exists(error_dir):
        #     os.makedirs(error_dir)
        # if not os.path.exists(self.plot_path_dir):
        #     os.makedirs(self.plot_path_dir)
        # if not os.path.exists(self.plot_error_dir):
        #     os.makedirs(self.plot_error_dir)

        # evaluation
        i=0    
        # compute sequence errors
        seq_err = self.calc_sequence_errors(poses_gt, poses_result)
        self.save_sequence_errors(seq_err, result_dir/'sequence_errors.txt')

        # Compute segment errors
        avg_segment_errs = self.compute_segment_error(seq_err)

        # compute overall error
        ave_t_err, ave_r_err = self.compute_overall_err(seq_err)
        print("Sequence: " + str(i))
        print("Translational error (%): ", ave_t_err*100)
        print("Rotational error (deg/100m): ", ave_r_err/np.pi*180*100)
        ave_t_errs.append(ave_t_err)
        ave_r_errs.append(ave_r_err)

        # Compute ATE
        ate = self.compute_ATE(poses_gt, poses_result)
        seq_ate.append(ate)
        print("ATE (m): ", ate)

        # Compute RPE
        rpe_trans, rpe_rot = self.compute_RPE(poses_gt, poses_result)
        seq_rpe_trans.append(rpe_trans)
        seq_rpe_rot.append(rpe_rot)
        print("RPE (m): ", rpe_trans)
        print("RPE (deg): ", rpe_rot * 180 / np.pi)

        # Plotting
        self.plot_trajectory(poses_gt, poses_result, result_dir)
        self.plot_error(avg_segment_errs, result_dir)

        # Save result summary
        self.write_result(f, i, [ave_t_err, ave_r_err, ate, rpe_trans, rpe_rot])

        f.close()

        print("-------------------- For Copying ------------------------------")
        for i in range(len(ave_t_errs)):
            print("{0:.2f}".format(ave_t_errs[i]*100))
            print("{0:.2f}".format(ave_r_errs[i]/np.pi*180*100))
            # print("{0:.2f}".format(seq_ate[i]))
            # print("{0:.3f}".format(seq_rpe_trans[i]))
            # print("{0:.3f}".format(seq_rpe_rot[i] * 180 / np.pi))
