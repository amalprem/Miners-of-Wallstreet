import logo from './logo.svg';
import './App.css';
import {DisplayNewsData,GetStocks, DisplayCompleteData} from './responseTable';
import GridButtons from './responseTable';
import React from 'react';

var jsonData = [
  { data: "We show News sentiment here", date: "The date for the news" },
];
var bestTickers = ["AAPL", "AMZN", "GOOG"];

class App extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      tickersList: [],
      receivedData: [],
      stock: "",
      isLoading: false,
      showGridButtons: false  // add a loading state to your component
    };
    this.getStockDetails = this.getStockDetails.bind(this);
    this.handleClick = this.handleClick.bind(this);
    this.updateReceivedData = this.updateReceivedData.bind(this);
  }

  updateReceivedData(data) {
    this.setState({ receivedData: data, isLoading: false }); // set isLoading to false when data is received
  }

  async getStockDetails(e) {
    this.setState({ isLoading: true }); // set isLoading to true when fetching data
    await fetch("http://127.0.0.1:5000/select-tickers/DOW_JONES", {
      headers: {
        Accept: "application/json",
        'Content-Type': "application/json",
      },
      method: "GET",
    })
      .then((response) => response.json())
      .then(async (response) => {
        console.log(response);
        this.setState({
          tickersList: bestTickers,
          isLoading: false,
          showGridButtons: true // set isLoading to false when data is received
        });
      })
      .catch((err) => {
        console.log(err);
        this.setState({ isLoading: false }); // set isLoading to false when there's an error
      });
  }

  async handleClick(e) {
    const item = e.target.innerText;
    this.setState({ isLoading: true }); // set isLoading to true when fetching data
    await fetch(`http://127.0.0.1:5000/predict-stock/${item}`, {
      headers: {
        Accept: "application/json",
        "Content-Type": "application/json",
        mode: "cors",
      },
      method: "GET",
    })
      .then((response) => response.json())
      .then((response) => {
        console.log(response)
        this.updateReceivedData(response);
      })
      .catch((err) => {
        console.log(err);
        this.setState({ isLoading: false }); // set isLoading to false when there's an error
      });
  }

  render() {
    const tickers = this.state.tickersList;
    const dataReceived = this.state.receivedData
      ? this.state.receivedData
      : jsonData;
    const grid = tickers.map((item, index) => {
      return (        
          <button key={index} onClick={this.handleClick} >
            {item}
          </button>
      );
    });
    return (
      <div className="App">
        <h1>Ready to Mine Wall Street?</h1>
        <div>
          <button name="fetchStocks" onClick={this.getStockDetails}>
            Get Tickers!!
          </button>
        </div>
        {this.state.showGridButtons && (
          <div>
            <p>We found these high-performing stocks for you:</p>
            <div className="grid">{grid}</div>
          </div>
        )}
        {this.state.isLoading ? ( // conditionally render the spinner
          <div className="spinner"></div>
        ) : (
            <div>
              <DisplayCompleteData response={dataReceived} />
              {dataReceived.final_score === "green" && (
                <img src="https://media.giphy.com/media/3o6MbtRx8nFU2n5wZO/giphy.gif" alt="green image" />
              )}
              {dataReceived.final_score === "red" && (
                <img src="https://media.giphy.com/media/PDessAxtRRQha4h1bM/giphy.gif" alt="red image" />
              )}
            </div>
        )}
      </div>
    );
  }
}

// function onClickButton(e){
//   fetch('http://127.0.0.1:5000/prediction/', 
//       {
//         headers: {
//           'Accept': 'application/json',
//           'Content-Type': 'application/json'
//         },
//         method: 'POST',
//         body: JSON.stringify(formData)
//       })
//       .then(response => response.json())
//       .then(response => {
//         this.setState({
//           result: response.result,
//           isLoading: false
//         });
//       });
// }

export default App;
