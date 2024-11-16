import { useEffect, useState } from 'react'
import reactLogo from './assets/react.svg'
import viteLogo from '/vite.svg'
import './App.css'
import { Link } from 'react-router-dom'

function App() {
  //testovye attributes dlya json
  const [message, setMessage] = useState("");
  const [name, setName] = useState("");

  // example of using fetch koroche
  useEffect(() => {
    //zaprosy na api delaem cherez /api
    //response prevrashaem v json
    /*
    api возвращает что то такое(тестовый варик):
    {
      message: "hello world",
      name: "John Doe"
    }
    */
    fetch("/api/get-message")
    .then(response => response.json())
    .then(data => {
      setMessage(data.message);
      setName(data.name);
    });
  }, []);

  return (
    <>
      <div>
        <a href="https://vite.dev" target="_blank">
          <img src={viteLogo} className="logo" alt="Vite logo" />
        </a>
        <a href="https://react.dev" target="_blank">
          <img src={reactLogo} className="logo react" alt="React logo" />
        </a>
      </div>
      <h1>Vite + React</h1>
      <div className="card">
        <button>
          message is {message}, name is {name}
        </button>
        <p>
          Edit <code>src/App.jsx</code> and save to test HMR
        </p>
      </div>
      <p className="read-the-docs">
        Click on the Vite and React logos to learn more
      </p>
      <Link to="/page">emaaa ahuet' rabotaet (nazhat')</Link>
    </>
  )
}

export default App
