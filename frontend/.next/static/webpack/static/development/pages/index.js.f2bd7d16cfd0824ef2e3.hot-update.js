webpackHotUpdate("static/development/pages/index.js",{

/***/ "./pages/index.js":
/*!************************!*\
  !*** ./pages/index.js ***!
  \************************/
/*! exports provided: default */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "default", function() { return Index; });
/* harmony import */ var _babel_runtime_corejs2_helpers_esm_classCallCheck__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! @babel/runtime-corejs2/helpers/esm/classCallCheck */ "./node_modules/@babel/runtime-corejs2/helpers/esm/classCallCheck.js");
/* harmony import */ var _babel_runtime_corejs2_helpers_esm_createClass__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! @babel/runtime-corejs2/helpers/esm/createClass */ "./node_modules/@babel/runtime-corejs2/helpers/esm/createClass.js");
/* harmony import */ var _babel_runtime_corejs2_helpers_esm_possibleConstructorReturn__WEBPACK_IMPORTED_MODULE_2__ = __webpack_require__(/*! @babel/runtime-corejs2/helpers/esm/possibleConstructorReturn */ "./node_modules/@babel/runtime-corejs2/helpers/esm/possibleConstructorReturn.js");
/* harmony import */ var _babel_runtime_corejs2_helpers_esm_getPrototypeOf__WEBPACK_IMPORTED_MODULE_3__ = __webpack_require__(/*! @babel/runtime-corejs2/helpers/esm/getPrototypeOf */ "./node_modules/@babel/runtime-corejs2/helpers/esm/getPrototypeOf.js");
/* harmony import */ var _babel_runtime_corejs2_helpers_esm_inherits__WEBPACK_IMPORTED_MODULE_4__ = __webpack_require__(/*! @babel/runtime-corejs2/helpers/esm/inherits */ "./node_modules/@babel/runtime-corejs2/helpers/esm/inherits.js");
/* harmony import */ var react__WEBPACK_IMPORTED_MODULE_5__ = __webpack_require__(/*! react */ "./node_modules/react/index.js");
/* harmony import */ var react__WEBPACK_IMPORTED_MODULE_5___default = /*#__PURE__*/__webpack_require__.n(react__WEBPACK_IMPORTED_MODULE_5__);
/* harmony import */ var axios__WEBPACK_IMPORTED_MODULE_6__ = __webpack_require__(/*! axios */ "./node_modules/axios/index.js");
/* harmony import */ var axios__WEBPACK_IMPORTED_MODULE_6___default = /*#__PURE__*/__webpack_require__.n(axios__WEBPACK_IMPORTED_MODULE_6__);
/* harmony import */ var lodash__WEBPACK_IMPORTED_MODULE_7__ = __webpack_require__(/*! lodash */ "./node_modules/lodash/lodash.js");
/* harmony import */ var lodash__WEBPACK_IMPORTED_MODULE_7___default = /*#__PURE__*/__webpack_require__.n(lodash__WEBPACK_IMPORTED_MODULE_7__);
/* harmony import */ var _lib_model_generator__WEBPACK_IMPORTED_MODULE_8__ = __webpack_require__(/*! ../lib/model-generator */ "./lib/model-generator.js");





var _jsxFileName = "/Users/Sun/Projects/135_AdvancedCG/Research/pyrender_examples/frontend/pages/index.js";




var numFeatures = 10;
var serverUrl = 'http://localhost:5000/';

var Index =
/*#__PURE__*/
function (_React$Component) {
  Object(_babel_runtime_corejs2_helpers_esm_inherits__WEBPACK_IMPORTED_MODULE_4__["default"])(Index, _React$Component);

  function Index(props) {
    var _this;

    Object(_babel_runtime_corejs2_helpers_esm_classCallCheck__WEBPACK_IMPORTED_MODULE_0__["default"])(this, Index);

    _this = Object(_babel_runtime_corejs2_helpers_esm_possibleConstructorReturn__WEBPACK_IMPORTED_MODULE_2__["default"])(this, Object(_babel_runtime_corejs2_helpers_esm_getPrototypeOf__WEBPACK_IMPORTED_MODULE_3__["default"])(Index).call(this, props));
    _this.updateModelDebounced = lodash__WEBPACK_IMPORTED_MODULE_7___default.a.debounce(_this.updateModel, 250, {
      'maxWait': 1000
    });
    _this.state = {
      inputVec: lodash__WEBPACK_IMPORTED_MODULE_7___default.a.times(numFeatures, lodash__WEBPACK_IMPORTED_MODULE_7___default.a.constant(0))
    };
    return _this;
  }

  Object(_babel_runtime_corejs2_helpers_esm_createClass__WEBPACK_IMPORTED_MODULE_1__["default"])(Index, [{
    key: "componentDidMount",
    value: function componentDidMount() {}
  }, {
    key: "handleButtonClick",
    value: function handleButtonClick() {
      Object(_lib_model_generator__WEBPACK_IMPORTED_MODULE_8__["generateModel"])();
    }
  }, {
    key: "handleSliderChange",
    value: function handleSliderChange(index, value) {
      var inputVec = this.state.inputVec;
      inputVec.splice(index, 1, value / 100);
      this.setState({
        inputVec: inputVec
      });
      this.updateModelDebounced(); // TODO request image
    }
  }, {
    key: "updateModel",
    value: function updateModel() {
      var _this2 = this;

      console.log('updateModel', this.state.inputVec); // console.log()

      axios__WEBPACK_IMPORTED_MODULE_6___default.a.post(serverUrl, {
        input: this.state.inputVec
      }, {
        responseType: "blob"
      }).then(function (response) {
        var reader = new window.FileReader();
        reader.readAsDataURL(response.data);

        reader.onload = function () {
          var imageDataUrl = reader.result;

          _this2.setState({
            previewImg: imageDataUrl
          });

          console.log('got data url'); //imageElement.setAttribute("src", imageDataUrl);
        };
      });
    }
  }, {
    key: "render",
    value: function render() {
      var _this3 = this;

      var previewImg = this.state.previewImg;
      return react__WEBPACK_IMPORTED_MODULE_5___default.a.createElement("div", {
        __source: {
          fileName: _jsxFileName,
          lineNumber: 57
        },
        __self: this
      }, react__WEBPACK_IMPORTED_MODULE_5___default.a.createElement("h1", {
        __source: {
          fileName: _jsxFileName,
          lineNumber: 58
        },
        __self: this
      }, "Generative Design with Autoencoders"), react__WEBPACK_IMPORTED_MODULE_5___default.a.createElement("div", {
        __source: {
          fileName: _jsxFileName,
          lineNumber: 59
        },
        __self: this
      }, lodash__WEBPACK_IMPORTED_MODULE_7___default.a.range(numFeatures).map(function (i) {
        return react__WEBPACK_IMPORTED_MODULE_5___default.a.createElement("div", {
          key: i,
          __source: {
            fileName: _jsxFileName,
            lineNumber: 61
          },
          __self: this
        }, react__WEBPACK_IMPORTED_MODULE_5___default.a.createElement("input", {
          type: "range",
          min: "0",
          max: "100",
          class: "slider",
          onChange: function onChange(e) {
            _this3.handleSliderChange(i, e.target.value);
          },
          __source: {
            fileName: _jsxFileName,
            lineNumber: 62
          },
          __self: this
        }));
      })), previewImg && react__WEBPACK_IMPORTED_MODULE_5___default.a.createElement("img", {
        src: previewImg,
        __source: {
          fileName: _jsxFileName,
          lineNumber: 69
        },
        __self: this
      }), react__WEBPACK_IMPORTED_MODULE_5___default.a.createElement("button", {
        onClick: this.handleButtonClick.bind(this),
        __source: {
          fileName: _jsxFileName,
          lineNumber: 72
        },
        __self: this
      }, "Do something"));
    }
  }]);

  return Index;
}(react__WEBPACK_IMPORTED_MODULE_5___default.a.Component);



/***/ })

})
//# sourceMappingURL=index.js.f2bd7d16cfd0824ef2e3.hot-update.js.map