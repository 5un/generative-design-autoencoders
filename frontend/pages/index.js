import React from 'react'
import axios from 'axios'
import _ from 'lodash'
import { generateModel } from '../lib/model-generator'

const numFeatures = 10
const serverUrl = 'http://localhost:5000/'

export default class Index extends React.Component {

  constructor(props) {
    super(props)
    this.updateModelDebounced = _.debounce(this.updateModel, 250, { 'maxWait': 1000 })
    this.state = {
      inputVec: _.times(numFeatures, _.constant(0.5))
    }
  }

  componentDidMount() {
    
  }

  handleButtonClick() {
    generateModel()
  }

  handleSliderChange(index, value) {
    let inputVec = this.state.inputVec
    inputVec.splice(index, 1, value / 100)
    this.setState({ inputVec })
    this.updateModelDebounced()
    // TODO request image
  }

  updateModel() {
    console.log('updateModel', this.state.inputVec)
    // console.log()
    axios.post(serverUrl, { input: this.state.inputVec }, { responseType:"blob" })
      .then(response => {

          var reader = new window.FileReader();
          reader.readAsDataURL(response.data); 
          reader.onload = () => {

              var imageDataUrl = reader.result;
              this.setState({ previewImg: imageDataUrl })
              console.log('got data url')
              //imageElement.setAttribute("src", imageDataUrl);

          }
      });
  }

  render() {
    const { previewImg } = this.state 
    return (
      <div>
        <h1>Generative Design with Autoencoders</h1>
        <div>
        {_.range(numFeatures).map(i => (
          <div key={i}>
            <input type="range" min="0" max="100" class="slider" 
                    onChange={e => {this.handleSliderChange(i, e.target.value)}}/>
          </div>
        ))}
        </div>

        {previewImg &&
          <img src={previewImg} />
        }

        {/*
        <button onClick={this.handleButtonClick.bind(this)}>Do something</button>
        */}
      </div>
    )
  }

}