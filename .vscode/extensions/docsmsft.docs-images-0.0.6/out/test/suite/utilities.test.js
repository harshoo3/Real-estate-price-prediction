"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
const assert = require("assert");
const utilities_1 = require("../../src/utilities");
suite("Utilities testing", () => {
    test("getFileName function", () => {
        assert.equal(utilities_1.getFileName("C:\\path\\to\\image.png"), "image.png");
        assert.equal(utilities_1.getFileName("D:\\image.png"), "image.png");
        assert.equal(utilities_1.getFileName(null), "");
    });
    test("toHumanReadableString function", () => {
        assert.equal(utilities_1.toHumanReadableString(777), "777 B");
        assert.equal(utilities_1.toHumanReadableString(8888), "8.9 kB");
        assert.equal(utilities_1.toHumanReadableString(99999), "100.0 kB");
        assert.equal(utilities_1.toHumanReadableString(123457), "123.5 kB");
        assert.equal(utilities_1.toHumanReadableString(8765432), "8.8 MB");
        assert.equal(utilities_1.toHumanReadableString(35473476, false), "33.8 MiB");
    });
    test("calculatePercentReduction function", () => {
        assert.equal(utilities_1.calculatePercentReduction(346363, 2345), "99.32%");
        assert.equal(utilities_1.calculatePercentReduction(45678, 40319), "11.73%");
        assert.equal(utilities_1.calculatePercentReduction(100, 90), "10.00%");
        assert.equal(utilities_1.calculatePercentReduction(100, 900), "-800.00%");
    });
    test("resultToString function", () => {
        assert.equal(utilities_1.resultToString({ wasCompressed: false, wasResized: false, file: "file" }), "Unable to compress \"file\".");
        assert.equal(utilities_1.resultToString({ wasCompressed: true, wasResized: false, file: "file", before: "100 MB", after: "90 MB", reduction: "10%" }), "Compressed \"file\" from 100 MB to 90 MB, reduced by 10%.");
        assert.equal(utilities_1.resultToString({ wasCompressed: true, wasResized: true, file: "file", before: "100 MB", after: "90 MB", reduction: "10%" }), "Compressed (and resized) \"file\" from 100 MB to 90 MB, reduced by 10%.");
    });
});
//# sourceMappingURL=utilities.test.js.map