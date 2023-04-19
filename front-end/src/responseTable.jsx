import React from "react";

var stockList = []
// export function GetStocks(){
//     return(<>
//     <div>
//         <button name="fetchStocks" onClick={getStockDetails}>Get Tickers!!</button>
//     </div>
//     </>)
// }
let news_Data = []
class GridButtons extends React.Component {
    handleClick = (event) => {
      const item = event.target.innerText;
      fetch(`http://127.0.0.1:5000/predict-stock/${item}`)
        .then(response => response.json())
        .then(response => {
            console.log(response)
            //this.props.onDataReceived(response);
            news_Data = [response.news_output]
            console.log(news_Data)
        });
    }
  
    render() {
      const { items, itemsPerRow } = this.props;
  
      const grid = items.map((item, index) => {
        return (
          <button key={index} onClick={this.handleClick}>
            {item}
          </button>
        );
      });
  
      return <div className="grid">{grid}</div>;
    }
  }
  
export default GridButtons;
export function DisplayNewsData ({newsData}) {
    const displayData = newsData.map((info) => {
        if (info){
          return(<tr>
            <td>{info.date}</td>
            <td>{info.data}</td>
            </tr>
            )
        }
        else{
          return null
        }
    })
    return(
        <div>
            <table class="table table-striped">
                <thead>
                    <tr>
                    <th>Date</th>
                    <th>Records</th>
                    </tr>
                </thead>
                <tbody>
                    {displayData}
                </tbody>
            </table>
        </div>
    )
}
export function DisplayCompleteData({response}){
    const { final_score, model_pred, news_output, tweet_sentiment, stock } = response;
    if(response && news_output){
      return (
        <>
          <p>And so we have our predictions for {stock} saying...</p>
          <table>
            <tbody>
              <tr>
                <td>Final Score:</td>
                <td>{final_score}</td>
              </tr>
              <tr>
                <td>Model Prediction:</td>
                <td>{model_pred}</td>
              </tr>
              <tr>
                <td>News Output Data:</td>
                <td>{news_output.data}</td>
              </tr>
              <tr>
                <td>News Output Date:</td>
                <td>{news_output.date}</td>
              </tr>
              <tr>
                <td>Tweet Sentiment:</td>
                <td>{tweet_sentiment}</td>
              </tr>
            </tbody>
          </table>
        </>
      );
    }
    else{
      return null;
    }
}