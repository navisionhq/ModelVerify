//
//  ModelVerifyTests.swift
//  ModelVerifyTests
//
//  Created by Madhav Jha on 11/23/18.
//  Copyright © 2018 Madhav Jha. All rights reserved.
//

import XCTest
import Vision
import CoreML

class ModelVerifyTests: XCTestCase {
    var sourceDir : String!

    override func setUp() {
        sourceDir = "/" + #file.split(separator: "/").dropLast().dropLast().joined(separator: "/")
    }
    
    func getModelFilePath(filename : String) -> String {
        let path : String = sourceDir + "/data/models/" + filename
        return path
    }
    
    func getImageFilePath(filename : String) -> String {
        let path : String = sourceDir + "/data/images/" + filename
        return path
    }

    func testCanAccessModel() {
        let modelFilePath = getModelFilePath(filename: "ResNet50.mlmodel")
        let modelUrl = URL(fileURLWithPath: modelFilePath)
        let data = try? Data(contentsOf: modelUrl)
        XCTAssertNotNil(data)
    }
    
    func testCanAccessImage() {
        let imageFilePath = getImageFilePath(filename: "IMG_0.jpg")
        let imageUrl = URL(fileURLWithPath: imageFilePath)
        let data = try? Data(contentsOf: imageUrl)
        XCTAssertNotNil(data)
    }
    
    func testCanCompileModel() {
        let modelFilePath = getModelFilePath(filename: "ResNet50.mlmodel")
        let modelUrl = URL(fileURLWithPath: modelFilePath)
        guard let compiledUrl = try? MLModel.compileModel(at: modelUrl) else { XCTFail("Faile to compile"); return }
        guard let modelForTest = try? MLModel(contentsOf: compiledUrl) else { XCTFail("Failed to load compiled model"); return }
        guard let model = try? VNCoreMLModel(for: modelForTest) else { XCTFail("Failed to convert to VNCoreMLModel"); return }
        XCTAssertEqual(model.className, "VNCoreMLModel")
    }
    
    func testModelCanPredict() {
        // Get full paths of files
        let modelFilePath = getModelFilePath(filename: "ResNet50.mlmodel")
        let imageFilePath = getImageFilePath(filename: "IMG_0.jpg")
        
        // Load image
        let imageUrl = URL(fileURLWithPath: imageFilePath)
        guard let data = try? Data(contentsOf: imageUrl) else { XCTFail("Failed to load image"); return }
        let inputImage = CIImage(data: data)
        
        // Load and compile model
        let modelUrl = URL(fileURLWithPath: modelFilePath)
        guard let compiledUrl = try? MLModel.compileModel(at: modelUrl) else { XCTFail("Faile to compile"); return }
        guard let modelForTest = try? MLModel(contentsOf: compiledUrl) else { XCTFail("Failed to load compiled model"); return }
        guard let model = try? VNCoreMLModel(for: modelForTest) else { XCTFail("Failed to convert to VNCoreMLModel"); return }
        
        // Make request and verify
        let request = VNCoreMLRequest(model: model) { (finishedReq, err) in
            if err != nil {
                XCTFail("err: " + err.debugDescription)
            }
            guard let results = finishedReq.results as? [VNClassificationObservation] else { XCTFail("Failed to obtain classification observation"); return }
            
            let prediction = results.first!
            XCTAssertGreaterThan(prediction.confidence, 0.75)
            XCTAssertEqual(prediction.identifier, "analog_clock")
        }
        let handler = VNImageRequestHandler(ciImage: inputImage!)
        try? handler.perform([request])
    }
    
}
