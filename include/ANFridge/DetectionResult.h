//
// Created by aojoie on 4/3/2023.
//

#ifndef ANFRIDGE_DETECTIONRESULT_H
#define ANFRIDGE_DETECTIONRESULT_H

struct DetectionResult {
    int id;
    float confidence;
    float x1, y1, x2, y2; /// top left and bottom right coordinate
};

#endif//ANFRIDGE_DETECTIONRESULT_H
