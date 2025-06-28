import DetailedResult from "./DetailedResult"
import Welcome from "./Welcome"
import Loading from "./Loading"
import { useFormStatus } from "@/context/FormContext"
import { useSelector } from "react-redux"

const DataContainer = () => {

  const {isLoading} = useFormStatus()
  const {data} = useSelector((state:any) => state.globalState)

  if (data.type === 'tested') {
    return <Welcome />
  }

  if (isLoading) {
    return <Loading />
  }

  return (
    <main className="w-3/5 h-full flex flex-col justify-start items-start p-6">
        <h3 className="text-lg uppercase font-semibold bg-primary rounded px-6 text-white">hasil klasifikasi</h3>
        <h1 className=" text-3xl font-bold uppercase mt-3">Motif Kain {data?.label}</h1>
        <DetailedResult />
    </main>
  )
}

export default DataContainer