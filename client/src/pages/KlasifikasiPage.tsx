import FormContainer from "@/globals/FormContainer"
import Container from "@/components/Container"
import { predictModel } from "@/actions/predictActions"
import ImageInput from "@/components/ImageInput"
import DataContainer from "@/components/DataContainer"

const KlasifikasiPage = () => {
  return (
    <FormContainer action={predictModel}>
      <section className="min-h-[100vh] w-[100vw] flex items-center">
        <Container className="min-w-[80%] max-w-[80%] h-[70%] flex justify-between items-start gap-x-10">
            <div className="w-2/5 flex items-center justify-center flex-col">
              <ImageInput />
            </div>
            <DataContainer />
        </Container>
      </section>
    </FormContainer>
  )
}

export default KlasifikasiPage